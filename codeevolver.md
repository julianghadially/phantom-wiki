I'll analyze the codebase by examining the entry point file, related files, and README.

<tool_call>
{"name": "Read", "arguments": {"path": "src/program/phantomwiki_pipeline/PhantomWikiReActPipeline.py"}}
</tool_call>
<tool_response>
import dspy

from src.program.react_agent import PhantomWikiReActAgent

class PhantomWikiReActPipeline(dspy.Module):
    def __init__(self):
        self.agent = PhantomWikiReActAgent()

    def forward(self, question):
        result = self.agent(question=question)
        return result
</tool_response>

<tool_call>
{"name": "Glob", "arguments": {"pattern": "src/program/**/*.py"}}
</tool_call>
<tool_response>
src/program/phantomwiki_pipeline/PhantomWikiReActPipeline.py
src/program/react_agent/__init__.py
src/program/react_agent/PhantomWikiReActAgent.py
src/program/phantomwiki_pipeline/__init__.py
src/program/tools/__init__.py
src/program/tools/retrieve.py
</tool_response>

<tool_call>
{"name": "Read", "arguments": {"path": "src/program/react_agent/PhantomWikiReActAgent.py"}}
</tool_call>
<tool_response>
import dspy

from src.program.tools.retrieve import retrieve

class AnswerQuestion(dspy.Signature):
    """Answer questions with short factoid answers."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="often between 1 and 5 words")

class PhantomWikiReActAgent(dspy.Module):
    def __init__(self):
        self.agent = dspy.ReAct(AnswerQuestion, tools=[retrieve])

    def forward(self, question):
        result = self.agent(question=question)
        return result
</tool_response>

<tool_call>
{"name": "Read", "arguments": {"path": "src/program/tools/retrieve.py"}}
</tool_call>
<tool_response>
import dspy

from src.retriever.retriever import PhantomWikiRetriever

retriever = PhantomWikiRetriever()

def retrieve(query: str) -> str:
    """Retrieve passages from a knowledge base given a query."""
    results = retriever(query)
    return results
</tool_response>

<tool_call>
{"name": "Read", "arguments": {"path": "src/metric/metric.py"}}
</tool_call>
<tool_response>
import dspy

def phantomwiki_f1_feedback(example, pred, trace=None):
    answer = example.answer
    prediction = pred.answer

    f1 = dspy.evaluate.answer_exact_match(example, pred)

    return f1
</tool_call>
</tool_response>

<tool_call>
{"name": "Glob", "arguments": {"pattern": "src/retriever/**/*.py"}}
</tool_call>
<tool_response>
src/retriever/retriever.py
</tool_response>

<tool_call>
{"name": "Read", "arguments": {"path": "src/retriever/retriever.py"}}
</tool_call>
<tool_response>
import dspy

class PhantomWikiRetriever(dspy.Module):
    def __init__(self):
        self.retriever = dspy.Retrieve(k=3)

    def forward(self, query: str) -> str:
        results = self.retriever(query_or_queries=query)
        passages = "\n\n".join(results.passages)
        return passages
</tool_response>

<tool_call>
{"name": "Read", "arguments": {"path": "README.md"}}
</tool_call>
<tool_response>
# PhantomWiki ReAct Pipeline

This is a DSPy program for answering questions from the PhantomWiki dataset using a ReAct agent with retrieval capabilities.

## Overview

The pipeline uses a ReAct (Reasoning + Acting) agent to answer factoid questions by iteratively retrieving relevant passages from a knowledge base.

## Architecture

- **Pipeline**: `PhantomWikiReActPipeline` - Top-level DSPy module that wraps the agent
- **Agent**: `PhantomWikiReActAgent` - ReAct agent with retrieval tool
- **Tools**: `retrieve` - Retrieval function using `PhantomWikiRetriever`
- **Retriever**: `PhantomWikiRetriever` - DSPy retriever fetching top-3 passages

## Metric

The pipeline is evaluated using `phantomwiki_f1_feedback`, which currently computes exact match (via `dspy.evaluate.answer_exact_match`) between the predicted and gold answers.

## Dataset

The pipeline is designed for the PhantomWiki dataset, which contains factoid questions requiring multi-hop reasoning over a synthetic knowledge base.

</tool_response>

```
PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## ARCHITECTURE TITLE: 5-Step Hop-Chain Planning ReAct with search_wiki_deep (k=10/k=30)

## ARCHITECTURE SUMMARY:
The system is a DSPy-based Retrieval-Augmented Generation pipeline for answering multi-hop factoid questions from the PhantomWiki dataset. `PhantomWikiReActPipeline` (`src/program/phantomwiki_pipeline.py`) wraps `PhantomWikiReAct` (`src/program/phantomwiki_module.py`), which implements a structured 5-step hop-chain planning and traversal protocol via a `dspy.ReAct` loop with four tools: `search_wiki`, `search_wiki_deep`, `take_notes`, and `read_notes`.

The `AnswerQuestion` signature instructs the agent to (Step 0) write an explicit hop-chain plan before searching, (Step 1) classify anchor type (named-person vs attribute-value) and answer type, (Step 2) traverse hops one at a time saving intermediate results in notes, (Step 3) apply the final relation to all previous-hop entities, and (Step 4) verify against the hop plan before finishing. Thread-local storage ensures safe concurrent evaluation.

## ARCHITECTURE DESCRIPTION:
**Entry point:** `PhantomWikiReActPipeline` (`src/program/phantomwiki_pipeline.py`) is a `dspy.Module` that configures a ColBERTv2 retrieval model and GPT-4.1-mini LM, then delegates `forward(question)` to `PhantomWikiReAct` within a `dspy.context`.

**Module layer:** `PhantomWikiReAct` (`src/program/phantomwiki_module.py`) owns a `dspy.Retrieve(k=10)` retriever and a `dspy.ReAct` loop (max 50 iterations) bound to the `AnswerQuestion` signature and four tools. Notes state is stored in `threading.local()` for thread-safety during parallel evaluation. Each call to `forward` resets the notes workspace.

**Signature:** `AnswerQuestion` contains a 5-step structured prompt. Step 0 requires writing an explicit hop-chain plan (saved as 'hop_plan') before any searching. Step 1 distinguishes named-person anchors (single entity) from attribute-value anchors (many entities, use search_wiki_deep). Step 2 traverses hop by hop, saving 'hop_N_results' notes for each intermediate hop, with a critical rule against applying the final relation too early. Step 3 applies the final relation to all entities from the last intermediate hop. Step 4 verifies hop count and answer type against the plan before calling finish. Output is `list[str]` covering all correct answers.

**Tools:** `search_wiki(query)` retrieves k=10 passages via ColBERTv2. `search_wiki_deep(query)` retrieves k=30 passages for exhaustive attribute-value anchor discovery. `take_notes(key, note)` saves findings to a thread-local dict. `read_notes(key)` reads one or all notes.

**Data flow:** question → `PhantomWikiReActPipeline.forward` → `PhantomWikiReAct.forward` (reset notes) → `dspy.ReAct` loop (write hop_plan → classify anchor/answer type → hop-by-hop traversal with notes → apply final relation → verify → finish) → `dspy.Prediction(answer=list[str])`.

**Metric:** `phantomwiki_f1_feedback` uses `dspy.evaluate.answer_exact_match` for binary scoring of `pred.answer` vs `example.answer`.
```
