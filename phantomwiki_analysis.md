# PhantomWiki Evaluation Architecture Analysis

## Overview of `phantom_eval`

The `phantom_eval` package is the evaluation harness for PhantomWiki. It lives at
`src/phantom_eval/` in the [kilian-group/phantom-wiki](https://github.com/kilian-group/phantom-wiki) repo.

### Key files

| File | Role |
|------|------|
| `__main__.py` | CLI entry point. Loads dataset, constructs agent, runs inference, saves predictions. |
| `agents/__init__.py` | Factory function `get_agent()` mapping method names → agent classes. |
| `agents/common.py` | Abstract `Agent` base class, `RAGMixin` (retriever logic), `SCMixin` (self-consistency voting). |
| `agents/cot.py` | `CoTAgent`, `CoTSCAgent`, `CoTRAGAgent` — chain-of-thought variants. |
| `agents/react.py` | `ReactAgent`, `ActAgent`, and hybrid agents (`React_CoTSCAgent`, `CoTSC_ReactAgent`). |
| `agents/react_bm25.py` | `ReactBM25Agent` — a more scalable ReAct agent using BM25 for the Search tool. |
| `prompts.py` | All prompt templates (zeroshot, fewshot, CoT, ReAct) with few-shot examples. |
| `score.py` | Scoring functions: `exact_match`, `precision`, `recall`, `f1`. |
| `evaluate_utils.py` | Utilities for loading predictions, joining with QA pairs, computing aggregate metrics. |
| `utils.py` | Dataset loading from HuggingFace or local, `normalize_pred()`. |
| `constants.py` | `answer_sep = ","`, `inf_temperature_hi = 0.7`. |

### Agent hierarchy

```
Agent (abstract)
├── NshotAgent (zeroshot / fewshot)
│   └── NshotRAGAgent (zeroshot-rag / fewshot-rag)  [uses RAGMixin]
│   └── NshotSCAgent (zeroshot-sc / fewshot-sc)     [uses SCMixin]
├── CoTAgent (cot)
│   ├── CoTSCAgent (cot-sc)                         [uses SCMixin]
│   └── CoTRAGAgent (cot-rag)                       [uses RAGMixin]
├── ReactAgent (react)
├── ActAgent (act)
├── React_CoTSCAgent (react->cot-sc)
├── CoTSC_ReactAgent (cot-sc->react)
└── ReactBM25Agent (react-bm25)
```

---

## Question 1: Replacing the retriever with ColBERT

### Current retriever architecture

The retriever lives entirely inside `RAGMixin` (in `agents/common.py`). It supports three backends:

1. **BM25** — via FlashRAG's `BM25Retriever`. Requires a pre-built index and corpus (JSONL files).
2. **Dense** — via FlashRAG's `DenseRetriever` with a sentence-transformer embedding model.
3. **FAISS** — via LangChain's FAISS vectorstore, using an OpenAI-compatible embedding server (vLLM).

All three are hidden behind a single interface:

```python
def get_RAG_evidence(self, question: str) -> str:
    if self.retrieval_method in ["bm25", "dense"]:
        docs = self.retriever._search(question, num=self.retriever_num_documents, return_score=False)
        docs = [doc["contents"] for doc in docs]
    else:
        docs = [doc.page_content for doc in self.retriever.invoke(question)]
    return "\n================\n\n".join(docs)
```

### What would need to change for ColBERT

ColBERT is a **late-interaction** retriever (token-level similarity via MaxSim). To swap it in:

if the index is pre-built on a server, you'd point to the server URL via an HTTP client.

2. **RAGMixin.get_RAG_evidence**: Add a branch:
   ```python
   case "colbert":
       results = self.colbert.search(question, k=self.retriever_num_documents)
       docs = [r["content"] for r in results]
   ```

3. **If ColBERT is hosted on a remote server**, you'd replace the retriever object with an HTTP
   client that calls the server's search API. The `get_RAG_evidence` method stays the same
   structurally — it takes a question string and returns concatenated document text.

4. **No other files need to change.** The agents (`CoTRAGAgent`, `NshotRAGAgent`, etc.) only
   call `self.get_RAG_evidence(question)` and are retriever-agnostic.

5. **For the ReactAgent / ReactBM25Agent**, the situation is different. These agents don't use
   `RAGMixin` at all. Instead, they do retrieval inside `_step_observation()` using direct
   pandas lookups (`text_corpus["title"] == action_arg`) or BM25 search.
   To use ColBERT in the agentic setting, you'd modify the `Search` action handler in
   `_step_observation()` to call ColBERT instead of BM25.

### Summary of changes

| Component | Change needed |
|-----------|--------------|
| `RAGMixin.__init__` | Add `"colbert"` case for initialization |
| `RAGMixin.get_RAG_evidence` | Add `"colbert"` branch |
| `ReactBM25Agent._step_observation` | Replace BM25 search with ColBERT call (if using ColBERT in agentic mode) |
| `__main__.py` CLI args | Add `--retrieval-method colbert` option |
| Everything else | No changes needed |

---

## Question 2: The CoTRAG Agent — detailed analysis

### How CoTRAGAgent actually works

`CoTRAGAgent` inherits from both `CoTAgent` and `RAGMixin`. The entire class is remarkably simple:

```python
class CoTRAGAgent(CoTAgent, RAGMixin):
    def __init__(self, text_corpus, llm_prompt, cot_examples="",
                 embedding_model_name="whereisai/uae-large-v1",
                 retriever_num_documents=4, ...):
        CoTAgent.__init__(self, text_corpus, llm_prompt, cot_examples)
        RAGMixin.__init__(self, text_corpus, embedding_model_name,
                          retriever_num_documents, ...)

    def _build_agent_prompt(self, question):
        evidence = self.get_RAG_evidence(question)  # <-- THE ONLY DIFFERENCE FROM CoTAgent
        return self.combine_evidence_and_question(evidence, question)
```

Compare with the base `CoTAgent`:

```python
class CoTAgent(Agent):
    def _build_agent_prompt(self, question):
        evidence = get_all_evidence(self.text_corpus)  # <-- ALL articles concatenated
        return self.combine_evidence_and_question(evidence, question)
```

The **only** difference is where the evidence comes from:
- `CoTAgent`: concatenates ALL articles from the corpus into the prompt (in-context).
- `CoTRAGAgent`: retrieves the top-k articles using the retriever and puts only those in the prompt.

### How many queries are actually executed?

**Exactly one.** The flow is:

1. `_build_agent_prompt(question)` is called.
2. This calls `self.get_RAG_evidence(question)`, which calls `self.retriever._search(question, num=4)`.
3. The retriever (BM25 or dense) runs a **single** query using the raw question text.
4. It returns the top 4 documents.
5. These 4 documents are concatenated and injected into the CoT prompt as `{evidence}`.
6. The LLM sees the 4 documents + the question + few-shot examples, and generates an answer in **one turn**.

There is **no query decomposition, no iterative retrieval, no multi-hop retrieval logic**.

### Why is this "stupid"? (And why the paper acknowledges this)

You are correct that this is extremely limited. The paper itself explicitly acknowledges this
as a key finding (Section 5, "Evaluating Reasoning"):

> "RAG prompting techniques stunt reasoning performance across the board—F1 scores are near zero
> on questions with 5 or more reasoning steps as opposed to 15 steps for in-context prompting."

> "We attribute this to a core problem with RAG prompting: retrieving documents in the initial
> prompt before starting to answer the question, as opposed to reasoning through the question
> and retrieving documents dynamically."

The paper's point is precisely that **this naive RAG approach doesn't work** for multi-hop questions.
That's the whole reason they also test ReAct (agentic) — to show that dynamic, multi-step retrieval
is necessary. The CoTRAG agent is a **baseline to demonstrate the failure mode**, not a serious
attempt at solving the task.

Consider a question like:
> "Who is the nephew of the friend of the person whose hobby is birdwatching?"

This requires:
1. Find people whose hobby is birdwatching → need article about that person.
2. Find their friend → need the friend's article.
3. Find the nephew of the friend → need the sibling's article and their children.

A single BM25 query on the raw question text would match on "birdwatching" and maybe return
the right first article, but would never retrieve the friend's or nephew's articles because
their names don't appear in the question at all.

### Converting to a DSPy ChainOfThought agent

In DSPy, the CoTRAG agent would collapse to roughly:

```python
import dspy

class PhantomWikiCoTRAG(dspy.Module):
    def __init__(self, num_docs=4):
        self.retrieve = dspy.Retrieve(k=num_docs)
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)
```

That's it. The phantom_eval `CoTRAGAgent` is ~200 lines because it manually handles:
- Prompt template construction and formatting
- LLM API calls (via their custom `LLMChat` abstraction)
- Response parsing (regex to extract `<answer>...</answer>` tags)
- Retriever initialization (supporting 3 backends)
- Batch inference
- Error handling and logging
- Agent interaction history tracking

DSPy abstracts all of that away. The retriever, prompt construction, LLM call, and answer
parsing are all handled by the framework.

To properly configure DSPy for this:
```python
# Configure the retriever (e.g., ColBERT via dspy.ColBERTv2)
colbert = dspy.ColBERTv2(url="http://your-colbert-server:8893/api/search")
dspy.settings.configure(rm=colbert, lm=dspy.LM("openai/gpt-4.1-mini"))

# Or BM25:
# bm25 = dspy.retrievers.BM25(index=..., k=4)
# dspy.settings.configure(rm=bm25, lm=...)
```

---

## Question 3: The ReAct Agent — detailed analysis and DSPy comparison

### What's in the ReactAgent?

The `ReactAgent` implements the [ReAct pattern](https://arxiv.org/abs/2210.03629)
(Thought → Action → Observation loop). Here's the complete flow:

```
┌─────────────────────────────────────────────────┐
│  Initial prompt: system instructions +          │
│  few-shot ReAct examples + question             │
│                                                 │
│  Loop (up to max_steps=50):                     │
│                                                 │
│  1. _step_thought():                            │
│     LLM generates "Thought N: ..."              │
│     (stop at "Action" token)                    │
│                                                 │
│  2. _step_action():                             │
│     LLM generates "Action N: Tool[arg]"         │
│     (stop at "Observation" token)               │
│                                                 │
│  3. _step_observation():                        │
│     Parse action → execute tool → return result │
│     "Observation N: <result>"                   │
│                                                 │
│  Actions are one of:                            │
│    - RetrieveArticle[name] → exact title match  │
│    - Search[keyword] → substring match on all   │
│      articles, returns matching titles           │
│    - Finish[answer] → return final answer       │
│                                                 │
│  Everything appended to a "scratchpad" string   │
│  that grows with each step.                     │
└─────────────────────────────────────────────────┘
```

### Key architectural details

1. **Scratchpad pattern**: The entire conversation history (all Thoughts, Actions, Observations)
   is concatenated into a single growing string. Each LLM call sends the full prompt
   (instructions + examples + question + scratchpad + leading prompt like "Thought 3: ")
   as a single user message. This is NOT a multi-turn conversation — it's a single-turn
   prompt that grows with each step.

2. **Tools are extremely simple**:
   - `RetrieveArticle[name]`: Does an exact case-insensitive match on `text_corpus["title"]`.
     Returns the full article text. No retriever model involved.
   - `Search[keyword]`: Substring search on `text_corpus["article"]` column. Returns a
     numbered list of matching article titles. No retriever model involved.
   - `Finish[answer]`: Terminates the loop with the answer.

3. **ReactBM25Agent** is a more scalable variant that replaces the pandas operations with:
   - `Search[entity]`: First tries exact title match, then falls back to BM25 retrieval.
   - `Lookup[keyword]`: Sentence-level keyword search within the current article.
   - `Finish[answer]`: Same as above.

4. **No embedding model is used** in the base `ReactAgent`. It's pure keyword/title matching.
   This works because PhantomWiki articles have predictable titles (character names) and
   structured text.

### How this differs from DSPy's ReAct

DSPy's `dspy.ReAct` provides the same Thought/Action/Observation loop but with key differences:

| Aspect | phantom_eval ReactAgent | DSPy ReAct |
|--------|------------------------|------------|
| Prompt construction | Manual string concatenation of scratchpad | Automatic via signatures and demos |
| Tool definitions | Hardcoded 3 tools in `_step_observation()` | Declarative via `dspy.Tool` objects |
| LLM interaction | Custom `LLMChat` wrapper, manual stop sequences | Handled by DSPy LM abstraction |
| Response parsing | Regex-based (`parse_action` method) | Built-in structured output parsing |
| Few-shot examples | Manually written in `prompts.py` (~100 lines of examples) | Automatically selected via optimizers (or manually via `demos`) |
| History management | Growing scratchpad string | Internal trajectory management |
| Error handling | Manual try/catch with max_steps | Built-in with configurable retries |
| Lines of code | ~250 lines for ReactAgent alone | ~10-20 lines of user code |

### Reproducing with DSPy ReAct

```python
import dspy

# Define tools
def retrieve_article(name: str) -> str:
    """Retrieve the wiki article for a person by their exact name."""
    matches = articles_df[articles_df["title"].str.lower() == name.lower()]
    if len(matches) == 0:
        return "No article exists for the requested entity."
    return matches.iloc[0]["article"]

def search_corpus(keyword: str) -> str:
    """Search for all articles containing the keyword. Returns matching article titles."""
    matches = articles_df[articles_df["article"].str.lower().str.contains(keyword.lower())]
    if len(matches) == 0:
        return "No articles contain the requested attribute."
    titles = [f"({i+1}) {t}" for i, t in enumerate(matches["title"].tolist())]
    return "\n".join(titles)

# Create DSPy ReAct agent
react = dspy.ReAct(
    signature="question -> answer",
    tools=[
        dspy.Tool(retrieve_article, name="RetrieveArticle",
                  desc="Retrieve the article for a person by name"),
        dspy.Tool(search_corpus, name="Search",
                  desc="Search for articles containing a keyword"),
    ],
    max_iters=50,
)

# Run
result = react(question="Who is the nephew of the friend of David?")
print(result.answer)
```

That's approximately 20 lines of user code to replicate ~250 lines of the phantom_eval ReactAgent.

### What phantom_eval's extra code is doing

The bulk of the phantom_eval code handles concerns that DSPy manages internally:

- **`LLMChat` + `InferenceGenerationConfig`**: DSPy's `dspy.LM` handles this.
- **Stop sequence management**: DSPy handles action boundary detection internally.
- **Scratchpad string building**: DSPy manages the trajectory internally.
- **`Conversation` / `Message` types**: DSPy uses its own internal message format.
- **`agent_interactions` tracking**: DSPy provides `inspect_history()`.
- **Usage aggregation across steps**: DSPy tracks this internally.
- **`parse_action` regex**: DSPy's tool-calling uses structured parsing.
- **Batch/async orchestration in `__main__.py`**: Would need to be written separately for DSPy too, but `asyncio.gather` is straightforward.
- **Hybrid agents** (`React_CoTSCAgent`, `CoTSC_ReactAgent`): Fallback logic between two strategies. This is custom behavior you'd implement yourself in DSPy.

---

## Question 4: Program path and metric path for CodeEvolver

### What CodeEvolver needs

CodeEvolver operates on two paths:
1. **Program path** — a Python file that takes a question and returns an answer.
2. **Metric path** — a Python file with a function that scores the answer against ground truth.

### Can we forego the entire phantom_eval evaluation script?

**Yes, completely.** The phantom_eval codebase is a monolithic evaluation harness that bundles
together concerns we don't need:

| phantom_eval concern | Do we need it? |
|---------------------|---------------|
| Dataset loading from HuggingFace | No — we have local JSON files already |
| Agent class hierarchy (9 agent classes) | No — our program IS the agent |
| LLM abstraction (`LLMChat`, vLLM, OpenAI, Gemini) | No — DSPy handles this |
| Prompt template management | No — DSPy handles this |
| Batch inference orchestration | No — CodeEvolver runs one question at a time |
| Prediction file I/O (JSON serialization) | No — CodeEvolver manages its own state |
| `evaluate_utils.py` (pandas joins, caching) | No — we just need the F1 function |

The only thing we actually need from phantom_eval is the **F1 scoring logic**, which is
15 lines of code.

### Recommended file structure

```
phantom-wiki/
├── program.py              ← CodeEvolver edits this
├── metric.py               ← Standalone scoring function
├── generate.py              ← Existing: pw-generate command
├── generate_questions.py    ← Existing: samples train/val/test splits
├── output/
│   └── depth_10_size_1000000/
│       ├── articles.json        ← 1.3 GB corpus (retriever indexes this)
│       ├── questions.json       ← All 600 generated questions
│       ├── phantomwiki_train.json  ← 150 questions
│       ├── phantomwiki_val.json    ← 150 questions
│       └── phantomwiki_test.json   ← 300 questions
└── phantomwiki_analysis.md  ← This document
```

### What `program.py` should look like

The program is a DSPy module. CodeEvolver owns this file and evolves it.
It must expose a callable that takes a question string and returns an answer string.

```python
import dspy
import json
import pandas as pd

# --- Configuration ---
lm = dspy.LM("openai/gpt-4.1-mini")
dspy.configure(lm=lm)

# --- Load articles for tool use ---
articles_df = pd.read_json("output/depth_10_size_1000000/articles.json")

# --- Tool definitions ---
def retrieve_article(name: str) -> str:
    """Retrieve the wiki article for a person by their exact name."""
    matches = articles_df[articles_df["title"].str.lower() == name.lower()]
    if len(matches) == 0:
        return "No article exists for this person."
    return matches.iloc[0]["article"]

def search_articles(keyword: str) -> str:
    """Search all articles for a keyword. Returns matching titles."""
    mask = articles_df["article"].str.lower().str.contains(keyword.lower(), na=False)
    titles = articles_df.loc[mask, "title"].tolist()
    if not titles:
        return "No articles found."
    return "\n".join(f"({i+1}) {t}" for i, t in enumerate(titles[:20]))

# --- DSPy Program ---
react = dspy.ReAct(
    signature="question -> answer",
    tools=[retrieve_article, search_articles],
    max_iters=10,
)

def solve(question: str) -> str:
    """Entry point for CodeEvolver. Takes a question, returns a comma-separated answer."""
    result = react(question=question)
    return result.answer
```

### What `metric.py` should look like

The metric is self-contained. It needs no external dependencies beyond the standard library.

```python
def normalize(text: str, sep: str = ",") -> set[str]:
    """Normalize a prediction or ground truth string into a set of lowercase answers."""
    return {a.strip().lower() for a in text.split(sep) if a.strip()}

def f1_score(pred: str, true_answers: list[str], sep: str = ",") -> float:
    """
    Compute answer-level F1 between predicted answers and ground truth answers.
    
    Args:
        pred: Comma-separated predicted answers from the program.
        true_answers: List of all valid ground truth answers.
    
    Returns:
        F1 score between 0.0 and 1.0.
    """
    pred_set = normalize(pred, sep)
    true_set = normalize(sep.join(true_answers), sep)

    if not pred_set or not true_set:
        return 0.0

    tp = sum(1 for a in pred_set if a in true_set)
    precision = tp / len(pred_set)
    recall = sum(1 for a in true_set if a in pred_set) / len(true_set)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
```

### How CodeEvolver connects them

```python
import json
from program import solve
from metric import f1_score

# Load validation set
with open("output/depth_10_size_1000000/phantomwiki_val.json") as f:
    val_questions = json.load(f)

# Evaluate
scores = []
for q in val_questions:
    pred = solve(q["question"])
    score = f1_score(pred, q["answer"])
    scores.append(score)

avg_f1 = sum(scores) / len(scores)
print(f"Average F1: {avg_f1:.4f}")
```

---

## Question 5: Clone the repo vs. keep it simple?

### What we'd get from cloning the phantom-wiki repo

The [kilian-group/phantom-wiki](https://github.com/kilian-group/phantom-wiki) repo contains:

| Directory | Contents | Do we need it? |
|-----------|----------|---------------|
| `src/phantom_wiki/` | Universe generation (family trees, facts, articles, questions, Prolog) | Only if regenerating data — we already have our generated output |
| `src/phantom_eval/` | Full evaluation harness (agents, prompts, scoring, LLM wrappers) | **No** — replaced by `program.py` + `metric.py` |
| `tests/` | Tests for the generation and evaluation code | No |
| `scripts/` | Shell scripts for running experiments | No |
| `pyproject.toml` | Package config with heavy dependencies (pyswip, flashrag, langchain, vllm, etc.) | No — would add unnecessary complexity |

### What we'd lose by NOT cloning

1. **Data regeneration capability** — the `pw-generate` CLI command. But we already have
   our generated output in `output/depth_10_size_1000000/`. If we ever need to regenerate
   with different parameters, we can `pip install phantom-wiki` and run the command
   without cloning the repo.

2. **HuggingFace dataset loading** — `phantom_eval.utils.load_data()`. Not needed since
   we have local JSON files.

3. **Multiple agent baselines for comparison** — useful if running a research comparison,
   but CodeEvolver only needs to optimize ONE program, not run all baselines.

### Recommendation: Keep it simple

**Do NOT clone the repo.** Instead, create two standalone files:

- `program.py` — The DSPy program (CodeEvolver's target)
- `metric.py` — The F1 scoring function (~15 lines)

Reasons:

1. **Dependency hygiene**: The phantom-wiki repo pulls in `pyswip` (SWI-Prolog bindings),
   `flashrag`, `langchain`, `vllm`, `nltk`, and many others. Our setup only needs `dspy`
   and `pandas`.

2. **No coupling**: phantom_eval's agent classes, prompt templates, and LLM wrappers are
   all replaced by DSPy. Keeping them around creates confusion about what's actually being used.

3. **Regeneration is separate**: If you need to generate a new PhantomWiki instance (new seed,
   different size), just run `pip install phantom-wiki && pw-generate ...` as a one-off command.
   The generation code doesn't need to live in your working repo.

4. **Everything important is in the output**: The `articles.json` and the train/val/test
   question splits are all we need. These are data files, not code.

5. **The F1 metric is trivial**: It's a 15-line function with zero external dependencies.
   No need to import the entire `phantom_eval` package for it.

### What you lose (and why it's fine)

| Lost capability | Mitigation |
|----------------|------------|
| `pw-generate` for new instances | `pip install phantom-wiki` in a venv when needed |
| Baseline agent implementations | Not needed — CodeEvolver evolves its own program |
| Prolog query evaluation | Not needed — we evaluate natural language answers |
| HuggingFace dataset integration | Not needed — we use local JSON files |
| Plotting utilities | Not needed for optimization loop |

---

## Summary: Key takeaways for our evaluation setup

1. **The CoTRAG agent is intentionally naive** — a single retrieval call with 4 documents.
   It's a baseline to show RAG alone fails on multi-hop questions. Don't emulate this
   architecture expecting good results.

2. **The ReAct agent is the most capable** but still struggles. It dynamically retrieves
   articles step-by-step as it reasons through the chain. This is the right pattern for
   PhantomWiki's multi-hop questions.

3. **ColBERT integration is straightforward** — it only touches the retriever initialization
   and the `get_RAG_evidence()` / `_step_observation()` methods. Everything else is
   retriever-agnostic.

4. **DSPy dramatically reduces code** for all of these agents. The phantom_eval codebase is
   ~1500 lines across the agent files, doing things DSPy handles automatically. The equivalent
   DSPy code would be ~50 lines total for all three agent types (ZeroShot-RAG, CoT-RAG, ReAct).

5. **For our evaluation harness**, we should:
   - Use DSPy's `dspy.ReAct` with ColBERT as the retriever for the best results.
   - Implement the F1 scoring function from `score.py` (it's simple: set-based precision/recall).
   - The metric function takes `(pred_str, true_answer_list)` and returns an F1 score.
   - The "program" to be evaluated is the DSPy module (e.g., a `dspy.ReAct` or `dspy.ChainOfThought`).
