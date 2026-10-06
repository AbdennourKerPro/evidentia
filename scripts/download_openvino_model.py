"""Legacy Qwen downloader, not required or used by the current API pipeline."""

from __future__ import annotations

from huggingface_hub import snapshot_download

from app.settings import get_llm_model_path


# Historical constants stay here, never in the active OpenAI gateway.
LLM_MODEL_ID = "OpenVINO/Qwen2.5-7B-Instruct-int4-ov"
LLM_MODEL_REVISION = "3a6dc61d2f19f9591e585d251262c154db5640cb"
REQUIRED_MODEL_FILES = (
    "openvino_model.xml",
    "openvino_model.bin",
    "openvino_tokenizer.xml",
    "openvino_detokenizer.xml",
    "tokenizer_config.json",
)


def main() -> None:
    """Keep the historical command usable without importing a native runtime."""

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
