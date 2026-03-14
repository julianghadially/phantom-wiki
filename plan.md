# Plan: PhantomWiki Baseline Programs & Metrics for CodeEvolver

## Goal

Create a lightweight evaluation setup for CodeEvolver to optimize PhantomWiki question-answering programs. No dependency on the phantom-wiki evaluation harness — standalone `phantom_wiki_program.py` (the evolvable module), `pipeline.py` (entry point with ColBERT RM), `metric.py`, and a thin `evaluate.py` runner.

---

## 1. Metric (`metric.py`)

### Specification (matches the paper)

The paper uses **answer-level F1**: "we prompt the LLMs to predict all answers as a comma-separated list and measure correctness with the answer-level F1 score."

This naturally penalizes:
- **Incorrect answers** → hurts precision
- **Low recall** → hurts recall (missing valid answers)
- **Hallucinated answers** → hurts precision

#### Implementation

Two variants following the GEPA pattern from LangProBe:
- `phantomwiki_f1` — returns a float (for `dspy.Evaluate`)
- `phantomwiki_f1_feedback` — returns `ScoreWithFeedback` (for GEPA optimization)

```python
import dspy
from dspy.teleprompt.gepa.gepa import ScoreWithFeedback


def normalize(text: str) -> str:
    """Lowercase, strip whitespace."""
    return text.strip().lower()


def f1_score(prediction: str, ground_truth: list[str], sep: str = ",") -> float:
    pred_set = {normalize(a) for a in prediction.split(sep) if a.strip()}
    true_set = {normalize(a) for a in ground_truth}

    if not pred_set and not true_set:
        return 1.0
    if not pred_set or not true_set:
        return 0.0

    tp = len(pred_set & true_set)
    precision = tp / len(pred_set)
    recall = tp / len(true_set)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def phantomwiki_f1(gold, pred, trace=None):
    """Answer-level F1. Returns float for Evaluate."""
    gold_answers = gold.answer if isinstance(gold.answer, list) else [gold.answer]
    pred_answer = getattr(pred, "answer", str(pred))
    return f1_score(pred_answer, gold_answers)


def phantomwiki_f1_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Answer-level F1 with textual feedback for GEPA."""
    gold_answers = gold.answer if isinstance(gold.answer, list) else [gold.answer]
    pred_answer = getattr(pred, "answer", str(pred))
    score = f1_score(pred_answer, gold_answers)

    # Parse prediction into set for detailed feedback
    pred_set = {normalize(a) for a in pred_answer.split(",") if a.strip()}
    true_set = {normalize(a) for a in gold_answers}
    correct = pred_set & true_set
    missed = true_set - pred_set
    extra = pred_set - true_set

    feedback = f"Gold answers ({len(true_set)}): {gold_answers[:5]}{'...' if len(true_set) > 5 else ''}. "
    feedback += f"Predicted ({len(pred_set)}): '{pred_answer[:200]}'. "
    feedback += f"F1: {score:.2f}. "
    if correct:
        feedback += f"Correct: {list(correct)[:5]}. "
    if missed:
        feedback += f"Missed: {list(missed)[:5]}{'...' if len(missed) > 5 else ''}. "
    if extra:
        feedback += f"Extra (wrong): {list(extra)[:5]}. "

    return ScoreWithFeedback(score=score, feedback=feedback)
```

Key design decisions:
- **Set-based matching** — order doesn't matter, duplicates ignored
- **Normalize to lowercase** — "Al Treat" matches "al treat"
- `prediction` is a comma-separated string (the LLM's raw output)
- `ground_truth` is the list from the dataset's `"answer"` field
- No partial credit for close matches — exact string match only after normalization
- **`ScoreWithFeedback`** — the GEPA variant provides detailed feedback on correct/missed/extra answers so CodeEvolver can understand *why* a score is low (precision vs recall issue)
- Follows the same `(gold, pred, trace)` signature convention as the HotpotQA metrics

---

## 2. Dataset

Already generated and split:

| File | Count | Purpose |
|------|-------|---------|
| `phantomwiki_train.json` | 150 | For DSPy optimizers (few-shot selection, prompt tuning) |
| `phantomwiki_val.json` | 150 | CodeEvolver's optimization target |
| `phantomwiki_test.json` | 300 | Final held-out evaluation |
| `articles.json` | 1,006,901 | The document corpus |

Question schema:
```json
{
  "id": "uuid",
  "question": "Who is the nephew of ...?",
  "answer": ["Name1", "Name2", ...],
  "difficulty": 7,
  "type": 2,
  "is_aggregation_question": false,
  "template": "...",
  "prolog": "...",
  "solution_traces": "..."
}
```

Observations from the data:
- Questions range from difficulty 1 to 10 (reasoning steps/hops)
- Many questions have **hundreds or thousands** of valid answers (e.g., 1923 answers for "date of birth of person whose hobby is table football")
- Aggregation questions (`is_aggregation_question: true`) ask "how many" and typically have 1-2 numeric answers
- The corpus has ~1M articles, so in-context approaches are impossible — retrieval is mandatory

---

## 3. Baseline Programs

Each baseline is split into two files:
- **Program module** — The `dspy.Module` that CodeEvolver evolves. Uses `dspy.Retrieve` for retrieval.
- **Pipeline** — The entry point that sets up the retriever (`CountingRM` wrapping `dspy.ColBERTv2`), injects it via `dspy.context(rm=...)`, and calls the program.

### Utility: `counting_rm.py`

Wraps any RM to count retrieval calls (useful for cost tracking and analysis).

```python
import dspy

class CountingRM(dspy.Retrieve):
    def __init__(self, rm):
        super().__init__()
        self.rm = rm
        self.call_count = 0

    def forward(self, query_or_queries, k=None, **kwargs):
        self.call_count += 1
        return self.rm(query_or_queries, k=k, **kwargs)

    def reset_count(self):
        self.call_count = 0
```

### Baseline 1: Multi-Hop RAG

#### Program (`program_multihop_rag.py`)

Based on the HotpotMultiHop pattern from overview.md. A fixed 2-hop retrieve-then-read pipeline. This is the module CodeEvolver would evolve (if targeting this baseline).

```python
import dspy

class PhantomWikiMultiHop(dspy.Module):
    def __init__(self):
        self.k = 7
        self.retrieve = dspy.Retrieve(k=self.k)
        self.create_query_hop2 = dspy.ChainOfThought("question, summary_1 -> query")
        self.summarize1 = dspy.ChainOfThought("question, passages -> summary")
        self.summarize2 = dspy.ChainOfThought("question, context, passages -> summary")
        self.generate_answer = dspy.ChainOfThought(
            "question, summary_1, summary_2 -> answer"
        )

    def forward(self, question):
        # Hop 1: retrieve on raw question
        hop1_docs = self.retrieve(question).passages
        summary_1 = self.summarize1(question=question, passages=hop1_docs).summary

        # Hop 2: generate refined query, retrieve again
        hop2_query = self.create_query_hop2(question=question, summary_1=summary_1).query
        hop2_docs = self.retrieve(hop2_query).passages
        summary_2 = self.summarize2(
            question=question, context=summary_1, passages=hop2_docs
        ).summary

        # Answer
        return dspy.Prediction(
            answer=self.generate_answer(
                question=question, summary_1=summary_1, summary_2=summary_2
            ).answer
        )
```

**Expected weakness:** Fixed 2-hop pipeline can't handle questions requiring 5-10 reasoning steps. The paper shows CoT-RAG F1 drops to near zero beyond 5 hops.

**Purpose:** Provides a low baseline to emphasize weakness of RAG — even multi-hop RAG.

### Baseline 2: ReAct Agent

#### Program (`phantom_wiki_program.py`)

The primary program for CodeEvolver to evolve. Uses `dspy.Retrieve` for retrieval (not pandas), so the pipeline controls which RM backs it.

```python
import dspy

class PhantomWikiReAct(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=5)
        self.react = dspy.ReAct(
            signature="question -> answer",
            tools=[self.search_wiki],
            max_iters=10,
        )

    def search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus. Returns relevant passages."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def forward(self, question):
        result = self.react(question=question)
        return dspy.Prediction(answer=result.answer)
```

#### Pipeline (`pipeline.py`)

The entry point. Sets up `CountingRM(dspy.ColBERTv2(...))` and injects it as the RM before calling the program.

```python
import dspy
from counting_rm import CountingRM
from phantom_wiki_program import PhantomWikiReAct

COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"

class PhantomWikiReActPipeline(dspy.Module):
    def __init__(self):
        self.rm = CountingRM(dspy.ColBERTv2(url=COLBERT_URL))
        self.program = PhantomWikiReAct()

    def forward(self, question):
        with dspy.context(rm=self.rm):
            return self.program(question=question)
```

**Key properties:**
- `dspy.Retrieve` in the program automatically uses whatever RM is in `dspy.context`
- The pipeline injects `CountingRM(ColBERTv2)` so all retrieval goes through ColBERT
- `max_iters=10` allows multi-hop reasoning
- CodeEvolver evolves `phantom_wiki_program.py`, not `pipeline.py`

### What CodeEvolver Can Modify

Starting from Baseline 2 (ReAct), CodeEvolver is free to:
- Change the DSPy module type (ReAct → ChainOfThought → custom pipeline)
- Add reasoning steps, decomposition, or planning modules
- Modify prompts and signatures
- Change the number of search results, iterations, etc.
- Add new modules (e.g., query decomposition, answer aggregation, self-verification)
- Add/remove/modify tools (must use `dspy.Retrieve` for retrieval — the pipeline controls the backing RM)

**No constraints** on architecture — the only thing measured is F1 on the validation set.

---

## 4. Evaluation Runner (`evaluate.py`)

Thin script that connects the pipeline and metric. This is what CodeEvolver calls.

```python
import json
import sys
from pathlib import Path

import dspy
from pipeline import PhantomWikiReActPipeline
from metric import phantomwiki_f1

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
```

---

## 5. Retriever Setup

ColBERT is hosted at:
```
COLBERT_URL = "https://julianghadially--colbert-server-colbertservice-serve.modal.run/api/search"
```

The pipeline injects it via `dspy.context(rm=self.rm)` where `self.rm = CountingRM(dspy.ColBERTv2(url=COLBERT_URL))`. Programs use `dspy.Retrieve(k=...)` which automatically picks up the RM from context — they never reference ColBERT directly.

---

## 6. File Structure

```
phantom-wiki/
├── pipeline.py                 ← Entry point: Pipeline class (sets up RM, calls program)
├── phantom_wiki_program.py     ← CodeEvolver's target (ReAct module)
├── counting_rm.py              ← CountingRM utility
├── metric.py                   ← F1 scoring function
├── evaluate.py                 ← Runner connecting pipeline + metric + dataset
├── program_multihop_rag.py     ← Baseline 1 program (for comparison)
├── generate.py                 ← Existing: pw-generate command
├── generate_questions.py       ← Existing: splits questions into train/val/test
├── overview.md                 ← Project overview
├── phantomwiki_analysis.md     ← Architecture analysis
├── plan.md                     ← This file
└── output/
    └── depth_10_size_1000000/
        ├── articles.json
        ├── questions.json
        ├── phantomwiki_train.json
        ├── phantomwiki_val.json
        ├── phantomwiki_test.json
        ├── facts.pl
        └── timings.csv
```

---

## 7. Implementation Steps

### Step 1: Create `metric.py`
- Implement `f1_score(prediction, ground_truth)` with normalization
- Add `phantomwiki_f1(gold, pred)` — float for `dspy.Evaluate`
- Add `phantomwiki_f1_feedback(gold, pred)` — `ScoreWithFeedback` for GEPA
- Write a few inline tests to verify correctness

### Step 2: Create `counting_rm.py`
- Implement `CountingRM` wrapper that counts retrieval calls
- Delegates to any underlying RM

### Step 3: Create `phantom_wiki_program.py` (Baseline 2: ReAct)
- Define `PhantomWikiReAct(dspy.Module)` with `dspy.Retrieve` and `dspy.ReAct`
- Tools use `dspy.Retrieve` for search (not pandas)
- `forward(question)` returns `dspy.Prediction(answer=...)`

### Step 4: Create `pipeline.py`
- Define `PhantomWikiReActPipeline(dspy.Module)`
- Sets up `CountingRM(dspy.ColBERTv2(url=COLBERT_URL))`
- Injects RM via `dspy.context(rm=self.rm)` and calls the program
- Configure DSPy LM (gpt-4.1-mini or similar)

### Step 5: Create `evaluate.py`
- Import pipeline, instantiate it
- Call `pipeline(question=q["question"])` for each question
- Read `.answer` from the result
- Compute F1 per question and aggregate
- Report mean F1, F1 by difficulty, total retrieval calls

### Step 6: Create `program_multihop_rag.py` (Baseline 1, optional for comparison)
- Implement `PhantomWikiMultiHop(dspy.Module)` with `dspy.Retrieve`
- Can be wrapped in its own pipeline class if needed

### Step 7: Run baselines and record initial scores
- Run both baselines on `phantomwiki_val.json`
- Record F1 scores as the starting point for CodeEvolver

### Step 8: Connect to CodeEvolver
- Point CodeEvolver at:
  - **Module path:** `phantom_wiki_program.py` (the `PhantomWikiReAct` class)
  - **Pipeline:** `pipeline.py` (wraps the program with ColBERT RM)
  - **Metric path:** `metric.py` (the `f1_score` function)
  - **Dataset:** `output/depth_10_size_1000000/phantomwiki_val.json`
- CodeEvolver evolves `phantom_wiki_program.py` to maximize mean F1

---

## 8. Expected Baselines (from paper)

For reference, the paper reports these F1 scores at n=5000 (our corpus is n≈1M, so expect lower):

| Method | GPT-4o F1 | Llama-3.3-70B F1 |
|--------|-----------|-------------------|
| ZeroShot | — (context too large) | — |
| CoT | — (context too large) | — |
| CoT-RAG | 6.96% | 8.67% |
| ReAct | 36.85% | 30.89% |

Our ReAct baseline should land somewhere in the 25-40% range. CodeEvolver's job is to push this higher through architectural and prompt evolution.

---

## 9. Key Considerations

### Answer Volume
Many questions have hundreds/thousands of valid answers. The system must:
- Generate comma-separated lists, not single answers
- Handle the LLM's tendency to stop after a few answers (recall penalty)
- This is a major optimization surface for CodeEvolver

### Difficulty Distribution
Questions span difficulty 1-10 (reasoning hops). Easy wins come from:
- Difficulty 1-3: direct lookups, should be near-perfect with good tools
- Difficulty 4-7: multi-hop, requires chaining lookups
- Difficulty 8-10: deep chains, where most systems fail

### Aggregation Questions
"How many X does Y have?" questions need counting logic. The answer is a number, not a list of names. These require different handling than entity-list questions.
