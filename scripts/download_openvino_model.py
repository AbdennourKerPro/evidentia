"""Download the pinned local OpenVINO LLM into Docker's persistent model volume."""

from __future__ import annotations

from huggingface_hub import snapshot_download

from app.llm_gateway import LLM_MODEL_ID, LLM_MODEL_REVISION, REQUIRED_MODEL_FILES
from app.settings import get_llm_model_path


def main() -> None:
    """Download all model files and verify that the OpenVINO pipeline can use them."""

    model_path = get_llm_model_path()
    print(f"Downloading {LLM_MODEL_ID}@{LLM_MODEL_REVISION} to {model_path}")

    snapshot_download(
        repo_id=LLM_MODEL_ID,
        revision=LLM_MODEL_REVISION,
        local_dir=model_path,
    )

    missing_files = [
        filename
        for filename in REQUIRED_MODEL_FILES
        if not (model_path / filename).is_file()
    ]
    if missing_files:
        raise RuntimeError(f"Model download is incomplete: {missing_files}")

    print("OpenVINO model download completed successfully.")


if __name__ == "__main__":
    main()
