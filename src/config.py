"""Configuration management for the video search and summarization service.

Loads settings from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NIMConfig:
    """NVIDIA Inference Microservice (NIM) endpoint configuration."""

    vlm_model: str = field(
        default_factory=lambda: os.environ.get(
            "VLM_MODEL_NAME", "nvidia/llava-v1.6-mistral-7b"
        )
    )
    llm_model: str = field(
        default_factory=lambda: os.environ.get(
            "LLM_MODEL_NAME", "meta/llama-3.1-8b-instruct"
        )
    )
    embedding_model: str = field(
        default_factory=lambda: os.environ.get(
            "EMBEDDING_MODEL_NAME", "nvidia/nv-embedqa-e5-v5"
        )
    )
    nim_base_url: str = field(
        default_factory=lambda: os.environ.get(
            "NIM_BASE_URL", "http://nim-proxy:8000/v1"
        )
    )
    api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("NVIDIA_API_KEY")
    )


@dataclass
class MilvusConfig:
    """Milvus vector database configuration."""

    host: str = field(
        default_factory=lambda: os.environ.get("MILVUS_HOST", "milvus")
    )
    port: int = field(
        default_factory=lambda: int(os.environ.get("MILVUS_PORT", "19530"))
    )
    collection_name: str = field(
        default_factory=lambda: os.environ.get(
            "MILVUS_COLLECTION", "video_embeddings"
        )
    )
    dim: int = field(
        default_factory=lambda: int(os.environ.get("EMBEDDING_DIM", "1024"))
    )


@dataclass
class VideoProcessingConfig:
    """Video ingestion and frame extraction settings."""

    frame_interval_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("FRAME_INTERVAL_SECONDS", "2.0")
        )
    )
    max_frames_per_video: int = field(
        default_factory=lambda: int(os.environ.get("MAX_FRAMES_PER_VIDEO", "500"))
    )
    supported_formats: list = field(
        default_factory=lambda: ["mp4", "avi", "mov", "mkv", "webm"]
    )
    output_dir: str = field(
        default_factory=lambda: os.environ.get("VIDEO_OUTPUT_DIR", "/tmp/frames")
    )
    upload_dir: str = field(
        default_factory=lambda: os.environ.get("VIDEO_UPLOAD_DIR", "/data/videos")
    )


@dataclass
class AppConfig:
    """Top-level application configuration."""

    host: str = field(
        default_factory=lambda: os.environ.get("APP_HOST", "0.0.0.0")
    )
    port: int = field(
        default_factory=lambda: int(os.environ.get("APP_PORT", "8000"))
    )
    debug: bool = field(
        default_factory=lambda: os.environ.get("APP_DEBUG", "false").lower() == "true"
    )
    log_level: str = field(
        default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO").upper()
    )
    nim: NIMConfig = field(default_factory=NIMConfig)
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    video: VideoProcessingConfig = field(default_factory=VideoProcessingConfig)


# Singleton config instance used across the application
config = AppConfig()
