"""Runtime configuration read from environment variables."""

import os
from pathlib import Path


def get_qdrant_url() -> str:
    """Return the internal Qdrant URL used by the API container."""

    return os.getenv("QDRANT_URL", "http://qdrant:6333")


def get_llm_model_path() -> Path:
    """Return the local directory containing the OpenVINO LLM files."""

    return Path(
        os.getenv(
            "LLM_MODEL_PATH",
            "/models/huggingface/qwen2.5-7b-instruct-int4-ov",
        )
    )


def get_openvino_device() -> str:
    """Return the explicitly configured OpenVINO inference device."""

    return os.getenv("OPENVINO_DEVICE", "CPU")
