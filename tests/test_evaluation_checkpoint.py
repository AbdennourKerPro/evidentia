"""Tests for resumable long-running agentic evaluations."""

from __future__ import annotations

import json
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

from scripts.evaluate_agentic_rag import PIPELINE_ID, _load_resumable_cases, source_fingerprint
from app.llm_gateway import get_llm_configuration
from app.retrieval_policy import get_retrieval_profile


class EvaluationCheckpointTests(unittest.TestCase):
    def write_checkpoint(self, path: Path, **overrides) -> None:
        payload = {
            "pipeline": PIPELINE_ID,
            "retrieval_profile": get_retrieval_profile(),
            "llm_configuration": get_llm_configuration(),
            "top_k": 5,
            "cases": [{"id": "case-a", "passed": True}],
        }
        payload.update(overrides)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_loads_cases_from_a_compatible_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(
                json.dumps(
                    {
                        "pipeline": PIPELINE_ID,
                        "retrieval_profile": get_retrieval_profile(),
                        "llm_configuration": get_llm_configuration(),
                        "cases": [{"id": "case-a", "passed": True}],
                    }
                ),
                encoding="utf-8",
            )
            loaded = _load_resumable_cases(path)
        self.assertEqual(loaded["case-a"]["passed"], True)

    def test_rejects_a_checkpoint_from_another_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(
                json.dumps({"pipeline": "old", "cases": []}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                _load_resumable_cases(path)

    def test_rejects_changed_model_or_reasoning_effort(self) -> None:
        for field, value in [("model_id", "old-model"), ("reasoning_effort", "high")]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "report.json"
                configuration = dict(get_llm_configuration(), **{field: value})
                self.write_checkpoint(path, llm_configuration=configuration)
                with self.assertRaisesRegex(ValueError, "LLM configuration"):
                    _load_resumable_cases(path)

    def test_rejects_changed_top_k(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            self.write_checkpoint(path)
            with self.assertRaisesRegex(ValueError, "top-k"):
                _load_resumable_cases(path, top_k=3)

    def test_rejects_changed_retrieval_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            self.write_checkpoint(path, retrieval_profile="different-profile")
            with self.assertRaisesRegex(ValueError, "retrieval profile"):
                _load_resumable_cases(path)

    def test_rejects_changed_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            benchmark = Path(directory) / "benchmark.jsonl"
            benchmark.write_text("original", encoding="utf-8")
            self.write_checkpoint(
                path, benchmark_sha256=sha256(benchmark.read_bytes()).hexdigest()
            )
            self.assertIn("case-a", _load_resumable_cases(path, benchmark=benchmark))
            benchmark.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "benchmark changed"):
                _load_resumable_cases(path, benchmark=benchmark)

    def test_rejects_changed_production_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            self.write_checkpoint(path, source_sha256={"app/candidate_recovery.py": "old"})
            with self.assertRaisesRegex(ValueError, "production source changed"):
                _load_resumable_cases(path)

    def test_accepts_matching_production_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            self.write_checkpoint(path, source_sha256=source_fingerprint())
            self.assertIn("case-a", _load_resumable_cases(path))


if __name__ == "__main__":
    unittest.main()
