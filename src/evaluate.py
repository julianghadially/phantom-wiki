import json
import sys
from pathlib import Path

import dspy
import mlflow
from src.program.phantomwiki_pipeline import PhantomWikiReActPipeline
from src.program.baseline_rag.baseline_pipeline import BaselineRAGPipeline
from src.program.baseline_rlm.rlm_pipeline import RLMPipeline
from src.metric.metric import phantomwiki_f1

dspy.configure(lm=dspy.LM("openai/gpt-4.1-mini", cache=False))

selected_pipeline = PhantomWikiReActPipeline
OPTIMIZED_PROGRAM_PATH = "codeevolver/results/optimized_program_20260326023331.json"


def _log_loaded_prompts(pipeline):
    """Log the instructions from each predictor so we can verify they were loaded."""
    for name, param in pipeline.named_parameters():
        if hasattr(param, "signature") and hasattr(param.signature, "instructions"):
            instructions = param.signature.instructions
            mlflow.log_text(instructions, f"prompts/{name}_instructions.txt")
            print(f"[mlflow] Logged prompt for {name} ({len(instructions)} chars)")


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

    pipeline = selected_pipeline()
    pipeline.load(OPTIMIZED_PROGRAM_PATH)
    print(f"Loaded optimized program from {OPTIMIZED_PROGRAM_PATH}")

    mlflow.set_experiment("phantomwiki-eval")
    with mlflow.start_run(run_name=f"eval-{split}-{Path(OPTIMIZED_PROGRAM_PATH).stem}"):
        mlflow.log_params({
            "split": split,
            "optimized_program": OPTIMIZED_PROGRAM_PATH,
            "pipeline": selected_pipeline.__name__,
            "max_questions": max_questions or "all",
        })

        # Log the full optimized program JSON as an artifact
        mlflow.log_artifact(OPTIMIZED_PROGRAM_PATH, "program")

        # Log each predictor's loaded instructions so we can verify they match
        _log_loaded_prompts(pipeline)

        scores = []
        by_difficulty = {}

        for i, q in enumerate(questions):
            print("--------------------------------")
            print(f"Question: {q['question']}")
            result = pipeline(question=q["question"])
            gold = dspy.Example(answer=q["answer"]).with_inputs("question")
            score = phantomwiki_f1(gold, result)
            scores.append(score)
            print(f"Result: {result.answer}")
            print(f"Answer: {q['answer']}")
            print(f"Score: {score}")

            mlflow.log_metric("f1", score, step=i)

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

        mlflow.log_metric("mean_f1", result["mean_f1"])
        mlflow.log_metric("total_retrieval_calls", result["total_retrieval_calls"])
        for d, f1 in result["f1_by_difficulty"].items():
            mlflow.log_metric(f"f1_difficulty_{d}", f1)

    return result


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "val"
    result = evaluate(split)
    print(json.dumps(result, indent=2))
