import json
import sys
from pathlib import Path

import dspy
from src.program.pipeline import PhantomWikiReActPipeline
from src.metric.metric import phantomwiki_f1


def evaluate(split: str = "val", max_questions: int | None = None) -> dict:
    split_file = {
        "train": "phantomwiki_train.json",
        "val": "phantomwiki_val.json",
        "test": "phantomwiki_test.json",
    }[split]

    data_path = Path("output/depth_10_size_1000000") / split_file
    with data_path.open() as f:
        questions = json.load(f)

    if max_questions:
        questions = questions[:max_questions]

    pipeline = PhantomWikiReActPipeline()

    scores = []
    by_difficulty = {}

    for q in questions:
        result = pipeline(question=q["question"])
        gold = dspy.Example(answer=q["answer"]).with_inputs("question")
        score = phantomwiki_f1(gold, result)
        scores.append(score)

        d = q["difficulty"]
        by_difficulty.setdefault(d, []).append(score)

    result = {
        "mean_f1": sum(scores) / len(scores) if scores else 0.0,
        "num_questions": len(scores),
        "f1_by_difficulty": {
            d: sum(s) / len(s) for d, s in sorted(by_difficulty.items())
        },
        "total_retrieval_calls": pipeline.rm.call_count,
    }
    return result


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "val"
    result = evaluate(split)
    print(json.dumps(result, indent=2))
