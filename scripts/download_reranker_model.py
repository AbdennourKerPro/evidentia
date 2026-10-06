"""Download the pinned multilingual cross-encoder into the model volume."""

from __future__ import annotations

from huggingface_hub import snapshot_download

from app.hybrid_retrieval import (
    CROSS_ENCODER_MODEL_ID,
    CROSS_ENCODER_MODEL_REVISION,
    CROSS_ENCODER_REQUIRED_FILES,
)
from app.settings import get_reranker_model_path


def main() -> None:
    """Download only the PyTorch and tokenizer files needed for CPU inference."""

    model_path = get_reranker_model_path()
    print(
        f"Downloading {CROSS_ENCODER_MODEL_ID}@{CROSS_ENCODER_MODEL_REVISION} "
        f"to {model_path}"
    )
    snapshot_download(
        repo_id=CROSS_ENCODER_MODEL_ID,
        revision=CROSS_ENCODER_MODEL_REVISION,
        local_dir=model_path,
        allow_patterns=[
            "*.json",
            "*.model",
            "model.safetensors",
            "README.md",
        ],
    )

    missing_files = [
        filename
        for filename in CROSS_ENCODER_REQUIRED_FILES
        if not (model_path / filename).is_file()
    ]
    if missing_files:
        raise RuntimeError(f"Reranker download is incomplete: {missing_files}")

    print("Cross-encoder download completed successfully.")


if __name__ == "__main__":
    main()
