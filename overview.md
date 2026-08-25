# Overview

## PhantomWiki
Paper: https://arxiv.org/pdf/2502.20377
Phantom wiki is an AI system benchmark for question answering with multi-branch reasoning and multi-hop retrieval.

Phantom Wiki creates a universe of fictional characters to evaluate reasoning and retrieval capabilities of language models and language model systems.

The benchmark is resilient against leakage because the facts are entirely fictional and random.

## Project Structure

```
src/
├── __init__.py
├── evaluate.py                         # Evaluation harness (entry point)
├── metric/
│   ├── __init__.py                     # Re-exports: normalize, f1_score, phantomwiki_f1, phantomwiki_f1_feedback
│   └── metric.py                       # F1 scoring + GEPA feedback metric
└── program/
    ├── __init__.py                     # Re-exports: CountingRM, PhantomWikiReActPipeline, PhantomWikiReAct
    ├── counting_rm.py                  # Retrieval wrapper that counts calls
    ├── phantomwiki_module.py           # ReAct agent module (the program CodeEvolver evolves)
    ├── phantomwiki_pipeline.py         # Pipeline: wires retriever + module together
    └── baseline_rag/
        ├── __init__.py
        ├── program_multihop_rag.py     # Baseline 1: 2-hop chain-of-thought RAG
        └── baseline_pipeline.py        # Pipeline for baseline 1

data/                                   # the splits evaluate.py actually reads
├── phantomwiki_trainval_omitsuperlong.json   # 228 rows — the training pool
└── phantomwiki_test_omitsuperlong.json       # 208 rows — HELD OUT

output/depth_10_size_1000000/           # generation artifacts, git-ignored
├── articles.json                       # Generated PhantomWiki corpus
├── facts.pl                            # Prolog fact base
├── questions.json                      # All generated questions
└── timings.csv
```

Note the two directories: `output/` is what `generate.py` produced and is not
tracked; `data/` holds the evaluation splits and is. Retrieval at run time goes
to the remote ColBERT endpoint, so `output/` is not needed in order to evaluate.

`phantomwiki_trainval_omitsuperlong.json` is exactly the union of the earlier
`train_omitsuperlong` + `val_omitsuperlong` splits (113 + 115 rows), which is
why those two — and the pre-`omitsuperlong` `train`/`val` pair they came from —
are no longer kept: they duplicated rows already present here. It shares zero
questions with the test split.

## Evaluation Pipeline Architecture

The evaluation pipeline has three layers: **evaluate** → **pipeline** → **module**.

### 1. Entry point: `src/evaluate.py`
- Configures the LM globally via `dspy.configure(lm=dspy.LM("openai/gpt-5.4-nano"))`
- Loads question JSON from `output/depth_10_size_1000000/` for the requested split (train/val/test)
- Instantiates a pipeline (currently `PhantomWikiReActPipeline`)
- Iterates over questions, calls the pipeline, scores each with `phantomwiki_f1`
- Returns aggregate results: mean F1, F1 by difficulty level, total retrieval calls
- Run via: `python -m src.evaluate [split]`

### 2. Pipeline layer: `src/program/phantomwiki_pipeline.py`
- `PhantomWikiReActPipeline` is a `dspy.Module` that owns two things:
  - **Retriever**: `CountingRM` wrapping `dspy.ColBERTv2` pointed at a remote ColBERT server on Modal
  - **Program module**: `PhantomWikiReAct`
- On `forward(question)`, it sets the retriever into dspy context via `dspy.context(rm=self.rm)` and delegates to the program module
- The ColBERT URL: `https://julianghadially--colbert-server-phantom-wiki-colbertserv-75bf93.modal.run/api/search`

### 3. Program module: `src/program/phantomwiki_module.py`
- `PhantomWikiReAct` is the core reasoning module — **this is what CodeEvolver evolves**
- Uses `dspy.ReAct` with a `search_wiki` tool (wraps `dspy.Retrieve`)
- Signature: `question -> answer: list[str]`
- `max_iters=50` (max tool-calling iterations), `k=7` (documents per retrieval)

### Supporting components

**CountingRM** (`src/program/counting_rm.py`): A `dspy.Retrieve` subclass that wraps any retrieval model and counts how many times it is called. Used to track retrieval cost in evaluation results.

**Metric** (`src/metric/metric.py`):
- `phantomwiki_f1(gold, pred)` — Token-level F1 between predicted and gold answer lists. Handles both single-string and list answers. Used by `evaluate.py`.
- `phantomwiki_f1_feedback(gold, pred)` — Same F1 but returns a `ScoreWithFeedback` object (from `dspy.teleprompt.gepa`) with detailed textual feedback about correct/missed/extra answers. Used by CodeEvolver's GEPA optimizer.
- Normalization: lowercase + strip whitespace, then set-based precision/recall/F1.

### Key architectural notes
- **DSPy framework**: Everything is built on DSPy. Modules are `dspy.Module` subclasses. Retrieval uses `dspy.Retrieve` (resolved from context). LM calls go through DSPy's global LM config.
- **Separation of pipeline and module**: The pipeline owns infrastructure (retriever, LM config). The module owns reasoning logic. CodeEvolver only modifies the module, not the pipeline or retriever.
- **Remote retrieval**: The ColBERT index is hosted on Modal as a serverless endpoint. The retriever is not local.
- **Answer format**: Answers are `list[str]` — questions can have multiple correct answers (e.g., "list all siblings of X"). The F1 metric compares answer sets.
- **Difficulty tracking**: Questions have a `difficulty` field. Evaluation breaks down F1 by difficulty level.

## CodeEvolver

CodeEvolver optimizes one program at a time, by starting with the initial program and making changes to the prompts and the code (including context pipeline, tooling, AI modules, AI module graph, etc.).

This combines several mechanisms:
- **Optimizer algorithm:** A language model makes point mutations to the code base, over many iterations, and the best solution is selected, based on a dataset and a reward metric.
- **Coding agents**: Autonomous agents execute code changes that are requested by the optimizer.
- **Git branching:** A git process manages evolving code across many git worktrees
- **Sandboxing for security:** Coding agents are a big cyber risk without sandboxing, network policies, etc.

### CodeEvolver integration points in this repo
- **Module path**: `src/program/phantomwiki_module.py` — the file CodeEvolver mutates
- **Metric path**: `src/metric/metric.py` — provides `phantomwiki_f1_feedback` (with GEPA `ScoreWithFeedback`)
- **Dataset**: `output/depth_10_size_1000000/` — train/val/test JSON splits

## Programs

### Baseline 1: Chain of thought, multi-hop RAG
- Located in `src/program/baseline_rag/`
- `PhantomWikiMultiHop`: Fixed 2-hop retrieve-then-summarize pipeline with ChainOfThought modules
- Pipeline: `BaselineRAGPipeline` (same structure as ReAct pipeline, just swaps the module)
- Hop 1: retrieve on raw question → summarize; Hop 2: generate refined query → retrieve → summarize; then answer from both summaries
- k=7 documents per retrieval

### Baseline 2: ReAct agent (primary program)
- Located in `src/program/phantomwiki_module.py`
- `PhantomWikiReAct`: Uses `dspy.ReAct` with a `search_wiki` tool for iterative retrieval
- Up to 50 iterations of tool use, k=7 documents per retrieval
- This is the program that CodeEvolver evolves

### CodeEvolver Programs
CodeEvolver will modify the ReAct agent as a baseline.