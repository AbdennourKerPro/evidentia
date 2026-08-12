"""Local OpenVINO text-generation boundary for the Evidentia RAG service."""

from __future__ import annotations

from functools import lru_cache

# Import the OpenVINO runtime first. Besides making the dependency explicit,
# this loads its native runtime before the GenAI extension is imported.
import openvino  # noqa: F401
import openvino_genai as ov_genai

from app.settings import get_llm_model_path, get_openvino_device


LLM_MODEL_ID = "OpenVINO/Qwen2.5-7B-Instruct-int4-ov"
LLM_MODEL_REVISION = "3a6dc61d2f19f9591e585d251262c154db5640cb"
REQUIRED_MODEL_FILES = (
    "openvino_model.xml",
    "openvino_model.bin",
    "openvino_tokenizer.xml",
    "openvino_detokenizer.xml",
    "tokenizer_config.json",
)


class LlmModelUnavailableError(RuntimeError):
    """Raised when the local OpenVINO model has not yet been downloaded."""


def is_model_downloaded() -> bool:
    """Check the files required by OpenVINO without initializing the model."""

    model_path = get_llm_model_path()
    return model_path.is_dir() and all(
        (model_path / filename).is_file() for filename in REQUIRED_MODEL_FILES
    )


@lru_cache
def _get_pipeline() -> ov_genai.LLMPipeline:
    """Load the large model once, on the first generation request only."""

    if not is_model_downloaded():
        raise LlmModelUnavailableError(
            "The OpenVINO LLM is not downloaded. Run the model download command first."
        )

    return ov_genai.LLMPipeline(
        str(get_llm_model_path()),
        get_openvino_device(),
    )


@lru_cache
def _get_tokenizer() -> ov_genai.Tokenizer:
    """Load the tokenizer and its model-provided Qwen chat template once."""

    if not is_model_downloaded():
        raise LlmModelUnavailableError(
            "The OpenVINO LLM is not downloaded. Run the model download command first."
        )

    return ov_genai.Tokenizer(str(get_llm_model_path()))


def generate_chat(*, system_message: str, user_message: str) -> str:
    """Generate one deterministic response using Qwen's bundled chat template."""

    prompt = _get_tokenizer().apply_chat_template(
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        add_generation_prompt=True,
    )

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = 350
    config.do_sample = False
    config.repetition_penalty = 1.05

    return str(_get_pipeline().generate(prompt, config)).strip()
