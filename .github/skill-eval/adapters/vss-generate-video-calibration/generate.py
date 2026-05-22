#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate Harbor tasks for the vss-generate-video-calibration skill.

The vss-generate-video-calibration skill drives AutoMagicCalib (AMC) — it handles
both deploying the AMC microservice (`vss-auto-calibration` + `vss-auto-calibration-ui`
containers via `references/deploy-auto-calibration-service.md`) and running
calibration workflows against it (videos, RTSP, or sample-dataset modes).

This skill does NOT require a pre-deployed VSS base/alerts/lvs profile; the spec
has no `profile` field. The agent is responsible for bringing up AMC itself as
part of the trial.

## Spec layout

    skills/vss-generate-video-calibration/eval/auto-calibration.json

The spec declares 11 `expects` items (one per trial step). Each step is dispatched
as an independent Harbor task under:

    .github/skill-eval/datasets/vss-generate-video-calibration/auto-calibration/<platform>/step-<N>/

One platform today: `RTXPRO6000BW` (gpu_count=1 per spec).

## Directory layout per step

    <platform>/step-<N>/
        instruction.md          — PREAMBLE + query N + env notes
        task.toml               — task metadata (skill, platform, step_index, step_count)
        tests/test.sh           — delegates to generic_judge.py
        tests/generic_judge.py  — copied from .github/skill-eval/verifiers/
        tests/auto-calibration.json — full spec (judge reads checks for step N)
        solution/solve.sh       — oracle placeholder
        skills/vss-generate-video-calibration/   — full skill copy
        environment/Dockerfile  — FROM scratch (BrevEnvironment takes over)

Usage from the repository root:
    python3 .github/skill-eval/adapters/vss-generate-video-calibration/generate.py \\
        --output-dir .github/skill-eval/datasets/vss-generate-video-calibration \\
        --skill-dir skills/vss-generate-video-calibration

    # One platform only:
    python3 .github/skill-eval/adapters/vss-generate-video-calibration/generate.py \\
        --output-dir .github/skill-eval/datasets/vss-generate-video-calibration \\
        --skill-dir skills/vss-generate-video-calibration \\
        --platform RTXPRO6000BW
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Platforms — mirrors the vss-manage-video-io-storage adapter convention
# ---------------------------------------------------------------------------

PLATFORMS: dict[str, dict] = {
    "H100":          {"short_name": "h100",          "gpu_type": "H100",         "min_vram_per_gpu": 80, "brev_search": "H100"},
    "L40S":          {"short_name": "l40s",          "gpu_type": "L40S",         "min_vram_per_gpu": 48, "brev_search": "L40S"},
    "RTXPRO6000BW":  {"short_name": "rtxpro6000bw",  "gpu_type": "RTX PRO 6000", "min_vram_per_gpu": 96, "brev_search": "RTX PRO"},
    "DGX-SPARK":     {"short_name": "spark",         "gpu_type": "GB10",         "min_vram_per_gpu": 96, "brev_search": "GB10"},
    "IGX-THOR":      {"short_name": "thor",          "gpu_type": "Thor",         "min_vram_per_gpu": 64, "brev_search": "Thor"},
}

# Prepended to every instruction.md so the skill's own HITL bypass
# clause fires. Skills default to "ask the user" before autonomous
# actions; in CI there's no user, so without this preamble the agent
# either stalls or falls through to a localhost default.
PREAMBLE = (
    "You are running inside a non-interactive evaluation harness. "
    "You are pre-authorized to deploy prerequisites autonomously — "
    "do not pause to ask for confirmation on `/vss-deploy-profile` or any other "
    "setup action the trial requires."
)

GENERIC_JUDGE = Path(__file__).resolve().parents[2] / "verifiers" / "generic_judge.py"


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def generate_test_script(step: int, spec_name: str) -> str:
    """Shell wrapper that invokes the generic LLM-as-judge verifier for a
    single step's checks. Harbor reads /logs/verifier/reward.txt."""
    return (
        "#!/bin/bash\n"
        f"# vss-generate-video-calibration verifier (step {step}): delegates to the\n"
        "# generic LLM-as-judge (.github/skill-eval/verifiers/generic_judge.py).\n"
        "set -uo pipefail\n"
        "\n"
        'TEST_DIR="$(cd "$(dirname "$0")" && pwd)"\n'
        "python3 -m pip install --quiet 'anthropic>=0.40.0' >/dev/null 2>&1 || true\n"
        "\n"
        'python3 "$TEST_DIR/generic_judge.py" \\\n'
        f'    --spec "$TEST_DIR/{spec_name}" --step {step}\n'
        "exit 0\n"
    )


def generate_solve_script(platform: str, step: int) -> str:
    """Oracle placeholder — the agent drives AMC; the verifier checks the
    result via curl / docker probes. There is no separate solve path."""
    return (
        "#!/bin/bash\n"
        f"# Gold solution placeholder: vss-generate-video-calibration step {step} on {platform}\n"
        "# The generic_judge verifier drives the checks via shell probes and\n"
        "# trajectory inspection. This solve.sh is a safety-net no-op.\n"
        "set -euo pipefail\n"
        "\n"
        "echo 'vss-generate-video-calibration: solve.sh is a no-op placeholder.'\n"
        "echo 'The generic_judge.py verifier performs all post-trial checks.'\n"
        "exit 0\n"
    )


def generate_task(
    platform: str,
    spec: dict,
    output_root: Path,
    skill_dir: Path,
    spec_stem: str,
) -> None:
    """Emit one Harbor task directory per entry in spec['expects'], i.e.
    step-<k>/ subdirs under `<spec_stem>/<platform_short>/`."""
    pspec = PLATFORMS[platform]
    platform_short = pspec["short_name"]
    expects = spec.get("expects") or []
    spec_name = f"{spec_stem}.json"

    for idx, expect in enumerate(expects, 1):
        step_dir = output_root / spec_stem / platform_short / f"step-{idx}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # ── instruction.md ────────────────────────────────────────────────
        # Replace {{platform}} placeholder in the query text
        query_raw = expect.get("query", "")
        query = query_raw.replace("{{platform}}", platform)
        env_text = (spec.get("env") or "").replace("{{platform}}", platform)

        lines = [
            PREAMBLE,
            "",
            f"## Trial step {idx} of {len(expects)} — {platform}",
            "",
            query,
            "",
            "## Environment notes",
            "",
            env_text,
            "",
            "Run autonomously without prompting for confirmation.",
            "",
        ]
        (step_dir / "instruction.md").write_text("\n".join(lines) + "\n")

        # ── task.toml ─────────────────────────────────────────────────────
        meta_lines = [
            "[task]",
            f'name = "nvidia-vss/vss-generate-video-calibration-{spec_stem}-{platform_short}-step-{idx}"',
            f'description = "AMC calibration step {idx}/{len(expects)} on {platform}"',
            f'keywords = ["vss-generate-video-calibration", "amc", "{platform}"]',
            "",
            "[environment]",
            'skills_dir = "/skills"',
            "",
            "[verifier.env]",
            'ANTHROPIC_API_KEY = "${ANTHROPIC_API_KEY}"',
            'ANTHROPIC_BASE_URL = "${ANTHROPIC_BASE_URL}"',
            # ANTHROPIC_MODEL gives the verifier's judge model cascade a
            # working fallback when JUDGE_MODEL is unset.
            'ANTHROPIC_MODEL = "${ANTHROPIC_MODEL}"',
            "",
            "[metadata]",
            'skill = "vss-generate-video-calibration"',
            # No `profile` field — this skill has no /vss-deploy-profile prerequisite.
            # BrevEnvironment's _ensure_prerequisite_deployed skips when absent.
            f'platform = "{platform}"',
            f'gpu_type = "{pspec["gpu_type"]}"',
            f'brev_search = "{pspec["brev_search"]}"',
            f'min_vram_gb_per_gpu = {pspec["min_vram_per_gpu"]}',
            f'gpu_count = {spec.get("resources", {}).get("platforms", {}).get(platform, {}).get("gpu_count", 1)}',
            "requires_deployed_vss = false",
            # prerequisite_deploy_mode is intentionally absent — this spec
            # declares no prerequisite_deploy_mode, so the consumer
            # `_ensure_prerequisite_deployed` fires its "desired=''" clean
            # path (wipes stale containers) and the trial runs from a
            # guaranteed-clean state.
            f"step_index = {idx}",
            f"step_count = {len(expects)}",
            f"check_count = {len(expect.get('checks') or [])}",
            "",
        ]
        (step_dir / "task.toml").write_text("\n".join(meta_lines))

        # ── environment/ ──────────────────────────────────────────────────
        env_dir = step_dir / "environment"
        env_dir.mkdir(exist_ok=True)
        (env_dir / "Dockerfile").write_text("FROM scratch\n")

        # ── tests/ — wrapper + generic judge + spec ───────────────────────
        tests_dir = step_dir / "tests"
        tests_dir.mkdir(exist_ok=True)
        (tests_dir / "test.sh").write_text(generate_test_script(idx, spec_name))
        if GENERIC_JUDGE.exists():
            shutil.copy(GENERIC_JUDGE, tests_dir / "generic_judge.py")
        # Copy the full spec so the judge can read all checks for step idx
        spec_src = skill_dir / "eval" / spec_name
        if spec_src.exists():
            shutil.copy(spec_src, tests_dir / spec_name)
        else:
            (tests_dir / spec_name).write_text(json.dumps(spec, indent=2))

        # ── solution/ ─────────────────────────────────────────────────────
        solution_dir = step_dir / "solution"
        solution_dir.mkdir(exist_ok=True)
        (solution_dir / "solve.sh").write_text(
            generate_solve_script(platform, idx)
        )

        # ── skills/ — full skill copy ─────────────────────────────────────
        dst = step_dir / "skills" / "vss-generate-video-calibration"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(skill_dir, dst)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Dataset output root "
             "(e.g. .github/skill-eval/datasets/vss-generate-video-calibration)",
    )
    parser.add_argument(
        "--skill-dir", required=True,
        help="Path to skills/vss-generate-video-calibration",
    )
    parser.add_argument(
        "--spec", default=None,
        help="Path to auto-calibration.json "
             "(default: <skill-dir>/eval/auto-calibration.json)",
    )
    parser.add_argument(
        "--platform", default=None,
        choices=list(PLATFORMS.keys()),
        help="Generate for this platform only "
             "(default: all platforms declared in spec resources.platforms)",
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    skill_dir = Path(args.skill_dir)
    spec_path = (
        Path(args.spec)
        if args.spec
        else skill_dir / "eval" / "auto-calibration.json"
    )

    if not spec_path.exists():
        print(f"spec not found: {spec_path}", file=sys.stderr)
        sys.exit(1)

    spec = json.loads(spec_path.read_text())
    spec_stem = spec_path.stem  # "auto-calibration"

    # Determine platforms from spec resources.platforms
    resources_platforms = (
        (spec.get("resources") or {}).get("platforms") or {}
    )
    if not resources_platforms:
        print(
            f"ERROR: spec {spec_path} is missing resources.platforms",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.platform:
        if args.platform not in resources_platforms:
            print(
                f"ERROR: platform {args.platform!r} not declared in spec "
                f"resources.platforms (declared: {list(resources_platforms)})",
                file=sys.stderr,
            )
            sys.exit(1)
        platforms = [args.platform]
    else:
        platforms = list(resources_platforms.keys())

    # Validate platforms
    for p in platforms:
        if p not in PLATFORMS:
            print(
                f"ERROR: platform {p!r} from spec is not in adapter PLATFORMS dict",
                file=sys.stderr,
            )
            sys.exit(1)

    print("=== Inputs ===")
    print(f"  output_dir   : {output_root}")
    print(f"  skill_dir    : {skill_dir}")
    print(f"  spec         : {spec_path}")
    print(f"  spec_stem    : {spec_stem}")
    print(f"  platforms    : {platforms}")
    print(f"  steps        : {len(spec.get('expects', []))}")
    print(
        f"  total checks : "
        f"{sum(len(q.get('checks', [])) for q in spec.get('expects', []))}"
    )
    print()

    for platform in platforms:
        task_id = PLATFORMS[platform]["short_name"]
        n_steps = len(spec.get("expects") or [])
        print(f"  GEN  vss-generate-video-calibration/{spec_stem}/{task_id}  ({n_steps} steps)")
        generate_task(platform, spec, output_root, skill_dir, spec_stem)

    print()
    print(f"Generated {len(platforms)} platform(s) × {len(spec.get('expects', []))} steps")
    print(f"  → {output_root}/{spec_stem}/")


if __name__ == "__main__":
    main()
