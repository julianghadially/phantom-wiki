PARENT_MODULE_PATH: src.program.phantomwiki_pipeline.PhantomWikiReActPipeline
METRIC_MODULE_PATH: src.metric.metric.phantomwiki_f1_feedback

## ARCHITECTURE TITLE: Three-Stage Pipeline with HopChainResolver Pre-pass, Two-Pass ReAct, and FinalAnswerSynthesizer Post-processing

## ARCHITECTURE SUMMARY:
The system is a DSPy-based question-answering pipeline targeting the PhantomWiki benchmark. `PhantomWikiReActPipeline` implements a three-stage strategy: a lightweight structured pre-pass (`HopChainResolver`), two sequential ReAct investigation passes (`PhantomWikiReAct` and `FollowUpInvestigation`), and a normalization post-processing step (`FinalAnswerSynthesizer`). The pre-pass decomposes multi-hop questions into ordered chain queries and resolves them via pure retrieval (no ReAct), producing `chain_candidates`. Both ReAct passes run as before, collecting initial and supplementary answers. The `FinalAnswerSynthesizer` ChainOfThought merges all three result sets, strips error strings, normalizes aggregation counts, and deduplicates semantically to produce the final answer.

`HopChainResolver` uses `HopChainExtractorSignature` (ChainOfThought) to decompose the question into up to 3 hop queries with `{hopN}` placeholders, then iterates: retrieving top-10 passages per hop and extracting up to 4 entity names via `EntityExtractorSignature` (ChainOfThought), substituting the first resolved entity into the next hop's query. The final hop's entities become `chain_candidates`. `FinalAnswerSynthesizerSignature` instructs the LLM to merge all sources, extract bare numeric counts for aggregation questions, remove null/error strings, and deduplicate.

## ARCHITECTURE DESCRIPTION:
**PhantomWikiReActPipeline** (`src/program/phantomwiki_pipeline.py`) is the top-level `dspy.Module`. On initialization it instantiates: a `CountingRM`-wrapped `dspy.ColBERTv2` retriever, a `HopChainResolver` for the pre-pass, a `PhantomWikiReAct` for pass 1, a `dspy.Retrieve(k=7)` for the follow-up search tool, a `dspy.ReAct` over `FollowUpInvestigation` for pass 2 (max_iters=25), and a `dspy.ChainOfThought(FinalAnswerSynthesizerSignature)` for final synthesis. The `forward` method runs all stages inside `dspy.context(rm=self.rm)`.

**HopChainResolver** (`src/program/phantomwiki_pipeline.py`) is a `dspy.Module` with three components: `dspy.ChainOfThought(HopChainExtractorSignature)` to decompose the question into hop queries, `dspy.Retrieve(k=10)` for retrieval-only search, and `dspy.ChainOfThought(EntityExtractorSignature)` to extract up to 4 entity names per hop. It iterates up to 3 hops, substituting resolved entities from hop N into hop N+1's query. The final hop's entities are returned as `chain_candidates`.

**HopChainExtractorSignature** decomposes a multi-hop question into an ordered `hops: list[str]` with `{hopN}` placeholder syntax. **EntityExtractorSignature** extracts up to 4 entity names from retrieved passages matching a hop query.

**FinalAnswerSynthesizerSignature** (`src/program/phantomwiki_pipeline.py`) is a DSPy Signature used by `dspy.ChainOfThought`. It takes `question`, `chain_candidates`, `pass1_answers`, and `pass2_answers` and outputs `answer: list[str]`. Its docstring instructs the LLM to: merge all sources, extract bare integer counts for aggregation questions (stripping "Name: N" format), remove error/null strings, and deduplicate semantically.

**FollowUpInvestigation** (`src/program/phantomwiki_pipeline.py`) is a DSPy Signature directing the second-pass ReAct agent to explore unexplored paths given already-found answers.

**PhantomWikiReAct** (`src/program/phantomwiki_module.py`) implements the primary ReAct agent with `dspy.Retrieve(k=7)` and up to 30 agentic iterations.

**CountingRM** (`src/program/counting_rm.py`) wraps any retriever and counts calls. **Metric — phantomwiki_f1_feedback** evaluates with set-based F1 and GEPA-compatible `ScoreWithFeedback`.

**Data flow**: question → `HopChainResolver` (decompose → retrieve k=10 → extract entities, up to 3 hops) → `chain_candidates` | `PhantomWikiReAct` (pass 1, up to 30 iters) → `pass1_answers` | `FollowUpInvestigation` ReAct (pass 2, up to 25 iters) → `pass2_answers` → `FinalAnswerSynthesizer` (merge + normalize + deduplicate) → `answer`.

## DSPy Patterns and Guidelines

DSPy is an AI framework for defining a compound AI system across multiple modules. Instead of writing prompts, we define signatures. Signatures define the inputs and outputs to a module in an AI system, along with the purpose of the module in the docstring. DSPy leverages a prompt optimizer to convert the signature into an optimized prompt, which is stored as a JSON, and is loaded when compiling the program.

**DSPy docs**: https://dspy.ai/api/

Stick to DSPy for any AI modules you create, unless the client codebase does otherwise.

Defining signatures as classes is recommended. For example:

```python
class WebQueryGenerator(dspy.Signature):
    """Generate a query for searching the web."""
    question: str = dspy.InputField()
    query: str = dspy.OutputField(desc="a query for searching the web")
```

Next, modules are used as nodes in the project, either as a single line:

```python
predict = dspy.Predict(WebQueryGenerator)
```

Or as a class:

```python
class WebQueryModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.query_generator = dspy.Predict(WebQueryGenerator)

    def forward(self, question: str):
        return self.query_generator(question=question)
```

A module can represent a single module, or the module can act as a pipeline that calls a sequence of sub-modules inside `def forward`.

Common prebuilt modules include:
- `dspy.Predict`: for simple language model calls
- `dspy.ChainOfThought`: for reasoning first, followed by a response
- `dspy.ReAct`: for tool calling
- `dspy.ProgramOfThought`: for getting the LM to output code, whose execution results will dictate the response

