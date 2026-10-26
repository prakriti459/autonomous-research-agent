import argparse
import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.database import SessionFactory, close_database
from app.db.models import Paper
from app.agent.workflow import answer_question


DEFAULT_DATASET = ROOT / "eval" / "questions.json"
DEFAULT_OUTPUT = ROOT / "eval" / "results.json"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).casefold()


def retrieval_hit(sources: list[dict[str, Any]], terms: list[str]) -> bool:
    normalized_terms = [_normalize(term) for term in terms]
    return any(
        all(term in _normalize(str(source.get("text", ""))) for term in normalized_terms)
        for source in sources
    )


async def run_evaluation(
    *, paper_id: str, dataset_path: Path
) -> dict[str, Any]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    questions = dataset.get("questions", [])
    if not 20 <= len(questions) <= 30:
        raise ValueError("Evaluation dataset must contain between 20 and 30 questions")

    results = []
    retrieval_hits = 0
    grounded_answers = 0
    total_latency_ms = 0.0
    async with SessionFactory() as db:
        paper = await db.get(Paper, paper_id)
        if paper is None:
            raise ValueError(f"Paper metadata not found for paper_id={paper_id}")

        for item in questions:
            started = perf_counter()
            result = await answer_question(
                db,
                question=item["question"],
                history=[],
            )
            latency_ms = (perf_counter() - started) * 1000
            hit = retrieval_hit(result.sources, item["expected_terms"])
            retrieval_hits += int(hit)
            grounded_answers += int(result.answer.grounded)
            total_latency_ms += latency_ms
            results.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "expected_answer": item["expected_answer"],
                    "answer": result.answer.answer,
                    "citations": [
                        citation.model_dump(exclude_none=True)
                        for citation in result.answer.citations
                    ],
                    "tool_used": result.tool_used,
                    "retrieval_hit": hit,
                    "grounded": result.answer.grounded,
                    "latency_ms": round(latency_ms, 2),
                }
            )

    total = len(questions)
    return {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "paper_id": paper_id,
        "paper_title": paper.title,
        "dataset_size": total,
        "retrieval_hit_rate": retrieval_hits / total,
        "grounded_answer_rate": grounded_answers / total,
        "average_latency_ms": total_latency_ms / total,
        "retrieval_hit_count": retrieval_hits,
        "grounded_answer_count": grounded_answers,
        "results": results,
    }


async def _run(args: argparse.Namespace) -> None:
    try:
        report = await run_evaluation(
            paper_id=args.paper_id,
            dataset_path=args.dataset,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
        print(f"Detailed results saved to {args.output}")
    finally:
        await close_database()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval hit rate, grounding rate, and latency."
    )
    parser.add_argument(
        "--paper-id",
        required=True,
        help="Paper UUID returned by POST /ingest",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
