import json
import logging
import sys
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import dspy
from src.program.phantomwiki_pipeline import PhantomWikiReActPipeline
from src.program.baseline_rag.baseline_pipeline import BaselineRAGPipeline
from src.program.baseline_rlm.rlm_pipeline import RLMPipeline
from src.metric.metric import phantomwiki_f1

# Suppress harmless asyncio event loop cleanup warnings from litellm/dspy threads
warnings.filterwarnings("ignore", message=".*coroutine.*was never awaited.*")
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

# Suppress BaseEventLoop.__del__ AttributeError spam from threads
import asyncio
_orig_del = asyncio.BaseEventLoop.__del__
def _quiet_del(self):
    try:
        _orig_del(self)
    except (AttributeError, Exception):
        pass
asyncio.BaseEventLoop.__del__ = _quiet_del

dspy.configure(lm=dspy.LM("openai/gpt-4.1-mini", cache=False))

selected_pipeline = PhantomWikiReActPipeline

NUM_THREADS = 4


def log_prompts(pipeline):
    print("=" * 60)
    print("LOADED PROMPTS / SIGNATURES")
    print("=" * 60)
    for name, predictor in pipeline.named_predictors():
        print(f"\n--- {name} ---")
        if hasattr(predictor, "signature"):
            sig = predictor.signature
            print(f"Instructions: {sig.instructions}")
            print(f"Fields: {[f.json_schema_extra['prefix'] for f in sig.fields.values()]}")
        if hasattr(predictor, "demos") and predictor.demos:
            print(f"Demos: {len(predictor.demos)}")
    print("=" * 60)


def evaluate(split: str = "val", max_questions: int | None = None, optimized_program: str | None = None) -> dict:
    split_file = {
        "train": "phantomwiki_train.json",
        "val": "phantomwiki_val.json",
        "test": "phantomwiki_test_omitsuperlong.json",
    }[split]

    data_path = Path("data") / split_file
    with data_path.open() as f:
        questions = json.load(f)

    if max_questions:
        questions = questions[:max_questions]

    pipeline = selected_pipeline()

    if optimized_program:
        with open(optimized_program) as f:
            state = json.load(f)
        pipeline.load_state(state)

    log_prompts(pipeline)
    print(f"\nRunning {len(questions)} questions with {NUM_THREADS} threads...\n")

    lock = threading.Lock()
    scores = []
    by_difficulty = {}

    def process_question(q):
        result = pipeline(question=q["question"])
        gold = dspy.Example(answer=q["answer"]).with_inputs("question")
        score = phantomwiki_f1(gold, result)
        print("--------------------------------")
        print(f"Question: {q['question']}")
        print(f"Result: {result.answer}")
        print(f"Answer: {q['answer']}")
        print(f"Score: {score}")
        with lock:
            scores.append(score)
            d = q["difficulty"]
            by_difficulty.setdefault(d, []).append(score)

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(process_question, q) for q in questions]
        for future in as_completed(futures):
            future.result()

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
    opt_program = sys.argv[2] if len(sys.argv) > 2 else None
    result = evaluate(split, optimized_program=opt_program)
    print(json.dumps(result, indent=2))
