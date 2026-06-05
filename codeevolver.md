I'll examine the relevant files to generate an accurate architecture summary.

<tool_call>
{"name": "Read", "arguments": {"path": "src/program/phantomwiki_pipeline/PhantomWikiReActPipeline.py"}}
</tool_call>
<tool_response>
import dspy
from src.program.phantomwiki_pipeline.retrieve import retrieve
from src.program.phantomwiki_pipeline.answer import AnswerQuestion

class PhantomWikiReActPipeline(dspy.Module):
    """Pipeline for answering questions from the PhantomWiki dataset."""
    def __init__(self):
        self.retrieve = retrieve
        self.answer = AnswerQuestion()

    def forward(self, question: str) -> dspy.Prediction:
        context = self.retrieve(question)
        answer = self.answer(context=context, question=question)
        return dspy.Prediction(answer=answer.answer)
</tool_response>

<tool_call>
{"name": "Glob", "arguments": {"pattern": "src/program/phantomwiki_pipeline/**/*"}}
</tool_call>
<tool_response>
src/program/phantomwiki_pipeline/retrieve.py
src/program/phantomwiki_pipeline/answer.py
src/program/phantomwiki_pipeline/__init__.py
</tool_call>
<tool_response>

<tool_call>
{"name": "Read", "arguments": {"path": "src/program/phantomwiki_pipeline/retrieve.py"}}
</tool_call>
<tool_response>
import dspy
from dspy.retrieve.you_rm import YouRM

def retrieve(query: str) -> list[str]:
    """Retrieve relevant context passages for the given query.
    
    Returns a list of text passages related to the query.
    """
    rm = dspy.settings.rm
    results = rm(query, k=3)
    
    if isinstance(results[0], dspy.Prediction):
        return [r.long_text for r in results]
    return results
</tool_call>
<tool_response>

<tool_call>
{"name": "Read", "arguments": {"path": "src/program/phantomwiki_pipeline/answer.py"}}
</tool_call>
<tool_response>
import dspy

class AnswerSignature(dspy.Signature):
    """Answer a question given relevant context passages from a knowledge base."""
    context: list[str] = dspy.InputField(desc="Relevant passages from a knowledge base")
    question: str = dspy.InputField(desc="Question to answer")
    answer: str = dspy.OutputField(desc="Answer to the question, as concise as possible")

class AnswerQuestion(dspy.Module):
    def __init__(self):
        self.generate_answer = dspy.ChainOfThought(AnswerSignature)
    
    def forward(self, context: list[str], question: str) -> dspy.Prediction:
        return self.generate_answer(context=context, question=question)
</tool_call>
<tool_response>

<tool_call>
{"name": "Read", "arguments": {"path": "src/metric/metric.py"}}
</tool_call>
<tool_response>
import re
import string
from dspy import Example

def normalize_answer(s: str) -> str:
    """Normalize an answer string for comparison."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punctuation(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punctuation(lower(s))))


def get_tokens(s: str) -> list[str]:
    """Get tokens from a normalized answer string."""
    if not s:
        return []
    return normalize_answer(s).split()


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth."""
    prediction_tokens = get_tokens(prediction)
    ground_truth_tokens = get_tokens(ground_truth)
    
    common = set(prediction_tokens) & set(ground_truth_tokens)
    num_common = sum(min(prediction_tokens.count(t), ground_truth_tokens.count(t)) for t in common)
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(prediction_tokens)
    recall = num_common / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def phantomwiki_f1_feedback(example: Example, prediction, trace=None) -> float:
    """Compute F1 feedback for PhantomWiki QA task."""
    answer = prediction.answer if hasattr(prediction, 'answer') else str(prediction)
    ground_truth = example.answer
    return compute_f1(answer, ground_truth)
</tool_call>
<tool_response>

<tool_call>
{"name": "Read", "arguments": {"path": "README.md"}}
</tool_call>
<tool_response>
# PhantomWiki ReAct Pipeline

A DSPy-based question answering pipeline for the PhantomWiki dataset.

## Overview

This project implements a retrieval-augmented generation (RAG) pipeline using DSPy to answer questions from the PhantomWiki dataset. The pipeline retrieves relevant context passages and uses a language model with chain-of-thought reasoning to generate answers.

## Architecture

The pipeline consists of two main components:

1. **Retrieval** (`src/program/phantomwiki_pipeline/retrieve.py`): Retrieves relevant passages from a knowledge base using DSPy's configured retrieval model
2. **Answer Generation** (`src/program/phantomwiki_pipeline/answer.py`): Uses DSPy's ChainOfThought to generate answers given retrieved context

## Metric

The pipeline is evaluated using token-level F1 score (`src/metric/metric.py`), which measures the overlap between predicted and ground truth answers after normalization.

## Setup

Install dependencies:
```
pip install dspy-ai
```

Configure DSPy with your language model and retrieval model:
```python
import dspy
lm = dspy.OpenAI(model="gpt-4")
rm = dspy.ColBERTv2(url="http://your-colbert-server")
dspy.configure(lm=lm, rm=rm)
```

## Usage

```python
from src.program.phantomwiki_pipeline import PhantomWikiReActPipeline

pipeline = PhantomWikiReActPipeline()
result = pipeline(question="What is the capital of France?")
print(result.answer)
```
</tool_call>
<tool_response>

```
PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## ARCHITECTURE TITLE: PhantomWiki RAG Pipeline with DSPy ChainOfThought and Token-Level F1 Optimization

## ARCHITECTURE SUMMARY:
The system is a retrieval-augmented generation (RAG) pipeline built with DSPy for question answering over the PhantomWiki dataset. It is composed of two sequential stages: a retrieval step (`src/program/phantomwiki_pipeline/retrieve.py`) that fetches relevant passages from a configured knowledge base, and an answer generation step (`src/program/phantomwiki_pipeline/answer.py`) that applies DSPy's `ChainOfThought` module to produce a concise answer from the retrieved context.

The top-level orchestrator (`PhantomWikiReActPipeline` in `src/program/phantomwiki_pipeline/PhantomWikiReActPipeline.py`) wires these two components together and exposes a single `forward(question)` interface. Performance is measured by `phantomwiki_f1_feedback` in `src/metric/metric.py`, which computes token-level F1 between the predicted and ground-truth answers.

## ARCHITECTURE DESCRIPTION:
**Entry point** — `src/program/phantomwiki_pipeline/PhantomWikiReActPipeline.py`
`PhantomWikiReActPipeline` is a `dspy.Module` subclass. Its `forward` method accepts a `question` string, calls `retrieve` to get context, passes both to `AnswerQuestion`, and returns a `dspy.Prediction` with a single `answer` field.

**Retrieval** — `src/program/phantomwiki_pipeline/retrieve.py`
The `retrieve` function reads the retrieval model from `dspy.settings.rm` (configurable at runtime — e.g., ColBERTv2 or YouRM) and fetches `k=3` passages for the query. It normalises the result format: if items are `dspy.Prediction` objects it extracts `long_text`; otherwise it returns the raw list.

**Answer generation** — `src/program/phantomwiki_pipeline/answer.py`
`AnswerQuestion` is a `dspy.Module` wrapping a `dspy.ChainOfThought` predictor bound to `AnswerSignature`. The signature declares `context` (list of passage strings) and `question` as inputs, and `answer` (as concise as possible) as the output field. ChainOfThought adds intermediate reasoning before emitting the final answer, making this step optimisable via DSPy's compilation/optimization loop.

**Metric** — `src/metric/metric.py`
`phantomwiki_f1_feedback(example, prediction)` normalises both the predicted answer (`prediction.answer`) and the ground-truth (`example.answer`) by lowercasing, removing punctuation and articles, then computes token-level precision and recall to produce an F1 score in [0, 1]. This scalar is used by the DSPy optimizer to tune the ChainOfThought prompts and/or retrieval parameters.

**Data flow**: `question` → `retrieve` (top-3 passages from knowledge base) → `AnswerQuestion.generate_answer` (ChainOfThought over context + question) → `answer` string → `phantomwiki_f1_feedback` compares against `example.answer` for optimization feedback.
```
