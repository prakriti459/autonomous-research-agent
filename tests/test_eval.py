import json

from eval.run_eval import retrieval_hit
from eval import run_eval


def test_sample_paper_evaluation_has_20_to_30_answered_questions() -> None:
    dataset = json.loads(run_eval.DEFAULT_DATASET.read_text(encoding="utf-8"))
    questions = dataset["questions"]

    assert 20 <= len(questions) <= 30
    assert all(
        item["question"] and item["expected_answer"] and item["expected_terms"]
        for item in questions
    )


def test_retrieval_hit_requires_expected_evidence_terms_in_one_source() -> None:
    sources = [
        {"text": "MOSS uses source-level adaptation for agent systems."},
        {"text": "The design keeps the batch bounded."},
    ]

    assert retrieval_hit(sources, ["source-level", "agent systems"])
    assert not retrieval_hit(sources, ["source-level", "batch bounded"])
