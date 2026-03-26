# CodeEvolver Memory
memory last updated at iteration number: 20

## Current Best Score
- **Best program**: Iteration 10 (adaptive two-pass ReAct with CompletenessCheck gating + deterministic normalizer, parent = candidate 3)
- **Best valset score**: 0.6083180805964966 (up from 0.5663 at iteration 9, +0.0421 improvement, all-time best)
- **Program**: Two-pass `dspy.ReAct` with `CompletenessCheck` (`dspy.ChainOfThought`) gating second pass, and `_normalize_answers` deterministic "Name — Value" stripper
- **Current program (iteration 20) valset score**: 0.5680431327752079 — below best (−0.0403 vs best); CompletenessCheck-gated architecture restored from iter 10 with enhanced `_normalize_answers`, parented off iter 10 (subsample parent 6.049)
- **Iteration 19 valset score**: 0.44354557843219217 — MAJOR REGRESSION (−0.1648 vs best); prompt mutation on react component parented off candidate 0 (subsample 5.613) instead of iteration 10
- **Iteration 18 valset score**: 0.5591469189437319 — below best (−0.0492); Entity Frontier Tracker parented off candidate 8 (subsample 5.137) instead of iteration 10
- **Iteration 16 valset score**: 0.43534398897516735 — MAJOR REGRESSION (−0.1730); targeted answer-completion loop parented off candidate 0 instead of iter 10

## Iteration History Summary

| Iteration | Change | Subsample Score | Valset Score | Delta (valset) |
|-----------|--------|----------------|-------------|----------------|
| 0 | Baseline vanilla dspy.ReAct | — | 0.44884 | — |
| 1 | Prompt mutation on react component | — | 0.51403 | +0.06519 (+14.5%) |
| 2 | 3-stage pipeline (QuestionDecomposer+ReAct+AnswerGapFinder) | 4.89 (↓ from 8.40) | None | — (rejected, large regression) |
| 3 | Empty change (no-op prompt on extract.predict) | 7.51 (flat) | None | — (no improvement) |
| 4 | Broad retrieval (k=50) + ExtractRequestedValues post-processor | 6.35 (↑ from 6.03) | 0.47432 | −0.0397 vs best |
| 5 | Prompt mutation on react component (parent=iter 1) | 9.09 (↑ from 7.89) | 0.52578 | +0.01175 (+2.3%) ← BEST |
| 6 | Add write_note/read_notes scratchpad tools to ReAct | 6.51 (↑ from 6.34, noise) | 0.44727 | −0.0785 vs best (REGRESSION) |
| 7 | Prompt mutation on extract.predict (parent=iter 0) | 8.42 (↑ from 6.92, +1.50) | 0.44714 | −0.0786 vs best (REGRESSION) |
| 8 | Replace LLM ExtractRequestedValues with deterministic "Name—Value" splitter (parent=iter 4 arch) | 9.98 (↑ from 8.11, +1.87) | 0.51110 | −0.0147 vs best |
| 9 | Prompt mutation on extract.predict (parent=candidate 3) | 10.528 (↑ from 10.479, +0.049) | 0.56626 | +0.0405 vs old best |
| 10 | Adaptive two-pass ReAct + CompletenessCheck + `_normalize_answers` (parent=candidate 3) | 11.926 (↑ from 9.114, +2.812) | 0.60832 | +0.0421 vs iter 9 ← **ALL-TIME BEST** |
| 16 | Targeted answer-completion loop (parent=candidate 0, baseline) | ~7.56 | 0.43534 | −0.1730 vs best (MAJOR REGRESSION) |
| 18 | Entity Frontier Tracker: 3-pass ReAct with EntityFrontierExtractor (parent=candidate 8, subsample 5.137) | 5.268 (↑ from 5.137, +0.131) | 0.55915 | −0.0492 vs best |
| 19 | Prompt mutation on react component (parent=candidate 0, subsample 5.613) | 6.538 (↑ from 5.613, +0.925) | 0.44355 | −0.1648 vs best (MAJOR REGRESSION) |
| 20 | Restore CompletenessCheck-gated 2-pass ReAct (from iter 10) + enhanced `_normalize_answers` (parent=iter 10) | 6.123 (↑ from 6.049, +0.074) | 0.56804 | −0.0403 vs best |

## What We Know About the Task

### Benchmark
- **PhantomWiki**: Multi-hop QA over fictional characters. Corpus hosted on ColBERT (remote Modal endpoint).
- Questions have difficulty levels 1–13 (number of reasoning hops).
- Metric: token-set F1 between predicted and gold answer lists (lowercased, set-based precision/recall).
- Answers are `list[str]` — questions can have multiple correct answers.

### Baseline Results (test set, 300 questions)
- **ReAct baseline** (current program): mean F1 = 0.3157, retrieval calls = 1636
- **RLM baseline** (large-context): mean F1 = 0.2485, retrieval calls = 6890
- Valset (current): 0.44884 — notably higher than test set, suggesting valset may have easier difficulty distribution

### Performance by Difficulty (ReAct baseline, test set)
- Difficulty 1: 0.471 (good)
- Difficulty 2: 0.249
- Difficulty 3: 0.625 (best!)
- Difficulty 4: 0.381
- Difficulty 5: 0.206
- Difficulty 6: 0.271
- Difficulty 7–9: 0.14–0.19
- Difficulty 10–11: 0.0 (complete failure)
- Difficulty 12: 0.285 (surprising recovery)
- Difficulty 13: 0.0 (complete failure)

**Key finding**: Performance collapses on high-difficulty (5+) questions. The program cannot reason across many hops reliably.

## What Works
- Basic ReAct loop with ColBERT retrieval is the best approach vs naive RAG or static RLM
- k=7 documents per retrieval seems sufficient for low-difficulty questions
- The RLM approach (large context, many retrievals) is WORSE — more retrieval calls doesn't help
- **Prompt mutation on the react component works well** — iteration 1 gained +14.5% and iteration 5 gained another +2.3% on valset, both purely via prompt changes
- Simple multi-hop name-based relationship traversal (e.g., 3-hop: find husband → find father → find sister) succeeds reliably in 2 searches
- The agent correctly handles single-answer and small-result set genealogy queries after the prompt mutation
- **Iterative prompt refinement on the same architecture (iter 1 → iter 5) continues to yield gains**: Subsample improved from 7.89 to 9.09 and valset from 0.51403 to 0.52578. Seven examples in iteration 5 achieved perfect scores (hobbies, occupations, counts, relationship lookups).
- **Prompt mutation on extract.predict CAN work when parent is strong**: Iteration 9 applied prompt mutation to `program.react.extract.predict` (same component as iteration 7 which failed), but achieved the new best valset score of 0.5663. The key difference vs iteration 7 appears to be parent selection — candidate 3 (subsample score ~10.48) vs iteration 0 (baseline). The right parent gives extract.predict mutations room to shine.
- **Two-pass ReAct with CompletenessCheck gating WORKS (iteration 10 = ALL-TIME BEST)**: The adaptive two-pass architecture lifted valset from 0.5663 to 0.6083 (+0.0421, +7.4%) and subsample from 9.11 to 11.93 (+30.8%). Key mechanics: (1) Pass 1 runs normally; (2) `CompletenessCheck` (ChainOfThought) decides if more searching is needed; (3) If incomplete, Pass 2 is run with context carrying over Pass 1 answers; (4) results are merged deduped. This is the first architectural change since iteration 0 to produce a major valset improvement.
- **CompletenessCheck gating successfully protects single-answer questions**: Single-answer examples (1, 2, 4, 6, 13, 14, 18) all stayed at 1.0 F1 — the completeness check correctly identified them as complete and skipped Pass 2. No regression on easy/single-answer questions.
- **Deterministic `_normalize_answers` ("Name — Value" stripper) is now confirmed integrated and working**: Part of the iteration 10 program. Correctly strips "Name — Value" format noise. Does NOT handle narrative answers like "Janey Hu has 1 friend." (only strips ` — ` separator).
- **Passing already-found answers in the second-pass context prompt is effective**: The follow-up query explicitly tells the agent what was already found, directing it to search for additional answers via different angles. This produced multi-entity improvements in examples 7, 8, 17, 19.

## What Doesn't Work
- The vanilla dspy.ReAct with no prompt optimization underperforms on multi-hop (3+ hop) questions
- Higher difficulty questions (10+) yield 0.0 F1 — the agent cannot follow long reasoning chains
- The RLM approach (baseline_rlm) failed despite 4x more retrieval calls — brute-force retrieval is not the solution
- **Aggregation/enumeration questions fail badly**: When a question requires finding ALL instances (e.g., "all people whose hobby is gongoozling"), the agent attempts to disambiguate or asks for clarification instead of exhaustively enumerating all matching entities. Example: hobby-based lookup returned a clarification question instead of searching all people with that hobby.
- **Incomplete descendant traversal**: For large family trees, the agent only explores a subset of branches and prematurely concludes no results exist — misses hundreds of valid answers.
- **Date-based entity lookup failures**: Agent cannot find people by date of birth (format mismatch between query and stored format in PhantomWiki).
- **Undercounting multi-hop chains**: Agent may count from the wrong intermediate node in a chain, or find only 1 valid answer when many are required.
- **Three-stage pipeline (iteration 2) was a major failure**: Rigid staged decomposition caused format contamination (entity-name-prefixed answers instead of bare values), lost self-correction behavior, and couldn't handle large answer sets. DO NOT re-attempt this approach without addressing those root causes.
- **Scratchpad tools (iteration 6) caused a major valset regression**: Adding `write_note`/`read_notes` tools to the ReAct tool list dropped valset from 0.52578 to 0.44727 (−0.0785). The agent never used the scratchpad tools in any observed trace — simply adding a tool to the tool list does NOT cause the model to use it. The subsample improved marginally (+0.167) but this was noise; the valset regression was severe. DO NOT add unused tools without either few-shot demonstrations or explicit prompt instructions that force their use.
- **Broad k=50 retriever (iteration 4) improved subsample but hurt valset**: Adding a broad retrieval tool helped on subsample but the overall valset score dropped vs. iteration 1. The k=50 tool can't cover large populations (hundreds/thousands of answers) and adds complexity without proportional gain.
- **Derived family relationships require graph traversal, not keyword search**: Second aunts, second cousins, etc. are computed by traversing the family graph (parent→siblings→their children). Searching for "second aunt" in wiki text returns nothing. The agent must walk: target → parents → grandparents → grandparents' siblings → filter by gender.
- **Missing parent links block reverse traversal**: Some entity wiki pages omit parent data entirely (e.g., Isidro Denney has no mother/father listed). This makes paternal-side family traversal impossible via forward lookup, causing undercounts in family relationship questions.
- **ExtractRequestedValues post-processor is beneficial but not enough**: The answer extractor correctly strips "Name — Value" format noise (confirmed fix on example 19), but the valset regression suggests it may introduce other errors elsewhere or that the broad retriever hurts more than the extractor helps.
- **LLM-based ExtractRequestedValues post-processor is actively harmful when used with correct ReAct outputs**: In iteration 8's subsample analysis, the LLM extractor was confirmed to drop 3 of 4 correct names (Examples 1/18), extract only 1 of 2 correct values (Examples 9/10), and reduce 4 found names to 1 (Examples 6/13). The LLM extractor is unreliable — it arbitrarily reduces answer sets even when ReAct got everything right. The deterministic "Name — Value" splitter fixed all these cases and improved subsample by +1.87 points.
- **Targeted answer-completion loop (iteration 16) was a major valset regression (−0.173)**: Despite subsample improvement (+0.88), valset dropped from 0.6083 to 0.4353. Root causes: (1) Parented off candidate 0 (baseline subsample 6.68) instead of the best program (iteration 10, subsample 11.93) — this alone may have caused most of the regression; (2) The attr_extractor (ChainOfThought) hallucinated false positive answers (e.g., "comptroller", "chief technology officer") that dilute F1; (3) Clarifying-question output from ReAct leaked into final answers, and the gap_finder couldn't recover; (4) 10-query cap still can't cover large answer sets (289 gold answers → 1 predicted); (5) Wrong initial anchor entities can't be corrected by targeted completion — the loop propagates wrong answers.
- **ChainOfThought attr_extractor introduces hallucinations**: When the LLM is asked to extract "new answers not already in existing_answers" from retrieval passages, it invents plausible but wrong values. LLM extractors over retrieved text are unreliable; prefer deterministic extraction or very tight grounding constraints.

## Major Opportunities

1. **Aggregation/enumeration handling (HIGHEST PRIORITY — still unsolved)**: Questions with 360–2,003 correct answers still score near 0.0 (examples 5, 16). Two passes only surface ~19 entities out of potentially thousands. The completeness checker's `follow_up_hint` is not enough to trigger exhaustive enumeration. A pagination or multi-query enumeration strategy is needed — e.g., loop through alphabet-initial queries, or re-query with different starting seeds until hitting diminishing returns.

2. **Narrative answer normalization**: The `_normalize_answers` method only strips "Name — Value" separators. It does NOT handle cases where the agent returns narrative strings like "Janey Hu has 1 friend." or "6 friends" when the expected answer is just the number. A regex-based numeric extractor for count questions (type 19 questions: "How many X does Y have?") could fix examples 9, 10 (both 0.0 F1).

3. **Missing entity returns error string (not empty list)**: When no entity matches (e.g., no person born on a specific date), the agent returns an error-message string instead of an empty list `[]`. This causes false positives. The program should catch "entity not found" cases and return `[]`.

4. **Further prompt mutation on top of iter 10 (HIGH VALUE, LOW RISK — MOST URGENT)**: Iteration 10 is still the all-time best (0.6083) after 10 subsequent iterations. Iterations 16, 18, and 19 all used wrong parents and regressed. Iteration 20 correctly used iter 10 as parent but scored 0.5680 instead of 0.6083. The SINGLE MOST IMPORTANT action is: use iteration 10 as parent AND ensure the code is a faithful reproduction of iteration 10's architecture. Prompt mutations on the `react` component, parented on iteration 10, are the safest next steps with the most headroom.

5. **Completeness checker improvements**: The completeness checker sometimes fails to recognize massive under-sampling (e.g., 2,003 correct answers but only 19 found). Improving the completeness check logic — e.g., adding a heuristic: "if answer count is small but question asks for a large population, flag as incomplete" — could improve Pass 2 trigger quality.

6. **Graph traversal for complex family relations**: Second-cousins-of-second-cousins-type questions are structurally impossible with current single-entry summaries. A dedicated multi-hop graph traversal tool (or wrapper that follows chains via multiple sequential lookups) would open up these question types.

7. **Better query formulation**: Instead of searching with partial facts, the agent could decompose the question into sub-questions and search for each entity specifically. An explicit query-decomposition step before ReAct would help.

8. **Increased k for later hops**: As the reasoning chain deepens, retrieving more documents (higher k) per query might help find the right entity article.

9. **Clarifying-question output filter**: When ReAct returns a string starting with "Which" or ending with "?" (or containing "do you mean"), treat the answer as `[]` and do not propagate it. This prevents format contamination in the final answer.

## Important Notes for Reflection Agent
- Do NOT change the retriever (ColBERT endpoint on Modal) — this is outside the module scope
- The evolvable file is `src/program/phantomwiki_module.py` only
- DSPy framework is used — use `dspy.ReAct`, `dspy.ChainOfThought`, `dspy.Predict`, etc.
- Signature must remain: `question -> answer: list[str]`
- The pipeline injects the retriever via `dspy.context(rm=self.rm)` — use `dspy.Retrieve(k=N)` inside the module
- `max_iters=50` is already set; the bottleneck is reasoning quality, not iteration count
- Prompt mutation alone gave a 14.5% valset improvement — further prompt refinement may still have headroom
- **Critical failure mode observed in traces**: Aggregation questions (find ALL X where attribute=Y then traverse relationships) return zero correct answers. The gold answers for aggregation questions include multiple distinct count values (e.g., ['0','1','2','3','4']) because there are multiple starting entities, each with a different count. The agent must find ALL starting entities first.
- **Subsample score does not reliably predict valset improvement**: Iteration 4 improved subsample (6.03 → 6.35) but hurt valset (0.514 → 0.474). Be cautious about changes that only improve the subsample without confirmed valset improvement.
- **The broad retriever (k=50) in iteration 4 is the suspected cause of valset regression**: It may retrieve noisy/irrelevant results that confuse reasoning on questions that were previously working. Consider reverting to single retriever k=7 or k=10 and just keeping the `ExtractRequestedValues` answer extractor.
- **ALL-TIME BEST is iteration 10**: Scored 0.6083 on valset, beating iteration 9's 0.5663. New changes MUST use iteration 10 as parent. Current iteration 16 scored only 0.4353 — it is a major regression.
- **Prompt mutation on both react and extract.predict components have now both succeeded**: Iterations 1 and 5 improved the react component; iteration 9 improved extract.predict. Both can yield valset gains with the right parent. Iteration 10 is the first architectural change to improve valset significantly (+7.4%). Scratchpad tools (iter 6) caused a large regression — tool-only changes without usage demonstrations still don't work.
- **Subsample score does not reliably predict valset**: Iteration 6 subsample improved slightly (6.34 → 6.51) while valset dropped 14.9%. Iteration 7 subsample improved substantially (6.92 → 8.42, +1.5) but valset was nearly unchanged. However, iteration 10 showed BOTH large subsample gain (+30.8%) AND large valset gain (+7.4%) — when the gain is very large in subsample, it's more likely to transfer. Iteration 16 showed subsample improved (+0.88) but valset dropped −0.173.
- **CRITICAL: Subsample improvement CANNOT be trusted if parent is wrong**: Iteration 16's subsample of 7.56 is far below iteration 10's 11.93 — it was parented off candidate 0 (baseline). The LLM comparison is between a weak baseline and a slightly-improved-weak-baseline. The "improvement" is illusory relative to the actual best program.
- **Adding tools to ReAct does NOT cause the model to use them**: The scratchpad tools in iteration 6 were added but never called in any trace. The model continued using only `search_wiki`. If adding tools, they MUST be accompanied by few-shot examples or explicit system prompt instructions that demonstrate/require their use.
- **Output format contamination is a persistent failure mode**: In iteration 6 traces, the agent returned "Carmine Libby — waste management officer" instead of just "waste management officer" for occupation questions. This "Name — Value" format breaks F1 scoring entirely. The `_normalize_answers` deterministic splitter in iteration 10 handles ` — ` separator but NOT narrative strings.
- **Premature "not found" hallucination is a persistent failure**: Agent concludes entity/relationship doesn't exist after a failed search and returns a verbose explanation string (score 0.0). Prompt should instruct: try multiple search strategies before concluding not found; never return explanation strings as answers. When no entity matches, return `[]` not an error string.
- **Date-of-birth (DOB) lookup is a confirmed systemic failure**: The agent cannot find entities by date of birth regardless of format attempted (ISO, spelled-out, European, US-style). Any question chain anchored on a birth date fails completely (score 0.0). The wiki search tool cannot resolve entities by DOB.
- **Prompt mutation on extract.predict result depends heavily on parent**: Iteration 7 applied the same type of mutation to `program.react.extract.predict` but got no valset improvement because it was parented off iteration 0 (baseline). Iteration 9 parented off a stronger candidate and achieved a new best. Always parent mutations off the best-performing candidate.
- **Parent selection matters CRITICALLY**: Changes should ALWAYS parent off the BEST program (iteration 10, valset 0.6083). Iteration 16 proved that parenting off a weak candidate (score 6.68) produces a weak program even if the architecture idea is sound.
- **Deterministic extractor is confirmed correct**: For "Name — Value" format, splitting on ` — ` and taking the right half works perfectly. For plain name lists, pass-through preserves all correct answers. This approach is strictly better than the LLM extractor.
- **Remaining failure modes in iteration 10 subsample**: (1) Narrative answer format not normalized (examples 9, 10 — "Janey Hu has 1 friend." instead of "1"); (2) Massive result-set undersampling (examples 5, 16 — 360–2,003 correct answers, only 0–19 found); (3) Complex relational chain traversal (cousin-of-cousin) structurally impossible with current tool; (4) Missing entity lookup returns error string instead of `[]`.
- **Iteration 20 subsample analysis confirms same structural failure patterns**: Low-scoring examples (IDs 1, 4, 5, 6, 8, 9, 13, 14, 17) all involve massive enumeration tasks with hundreds-to-thousands of gold answers. Partial recovery observed in examples 9, 11, 13 (Pass 2 found a few additional correct answers). False "not found" declarations in examples 2, 5, 8, 12, 14 — answer was sometimes briefly found in the trace then abandoned (Pass 2 didn't recover). Example 2 specifically: correct answer found mid-trace but discarded — "keep partial results" mechanic could recover these.
- **"date of birth" prefix stripping needed in `_normalize_answers`**: Iteration 20's enhanced normalizer strips "Name: value" colon patterns but NOT "date of birth XXXX-XX-XX." → "XXXX-XX-XX" patterns. This is a remaining format failure causing 0 F1 on some DOB questions.
- **Iteration 10 confirmed: both subsample and valset gained substantially**: subsample +30.8% (9.11→11.93), valset +7.4% (0.5663→0.6083). The two-pass ReAct architecture is now the confirmed best approach. Future iterations should build on this foundation.
- **Iteration 18 EntityFrontierExtractor architecture notes**: The design is theoretically sound — naming intermediate entities and providing targeted reverse-link queries is better than vague "try other angles" hints. However: (1) parented off wrong candidate; (2) ColBERT retrieval fails even on exact entity names, meaning more specific queries don't help when the index doesn't surface the document; (3) count questions answered with "6 friends" instead of "6" — `_normalize_answers` does not strip trailing unit words. If revisiting this architecture, must parent off iteration 10.
- **`_normalize_answers` does NOT strip trailing unit words**: The function only strips "Name — Value" separator format. Count/numeric answers like "6 friends", "3 children" are NOT normalized to "6" or "3". This causes F1=0 on count questions even when the underlying answer is correct. A targeted fix for type-19 questions ("How many X does Y have?") should extract the numeric part only.
- **Enhanced `_normalize_answers` (iteration 20) still does not fully normalize date/narrative strings**: Even with "Name: value" colon-pattern stripping and relationship-prefix stripping, answers like `"date of birth 0903-07-12."` and long explanatory strings remain unnormalized (example 10). Additional regex stripping for "date of birth" prefix and trailing periods is needed.
- **Iteration 20 restored CompletenessCheck but scored 0.5680, not 0.6083**: Despite being parented off iteration 10 and re-implementing its architecture, the valset score is −0.0403 below the original iteration 10. This suggests the enhanced `_normalize_answers` (the additional colon-pattern and relationship-prefix stripping) introduced some regressions, OR there are subtle implementation differences from the original iteration 10 code. The subsample parent score was only 6.049 (vs iteration 10's historic 11.926), suggesting the subsample used for evaluation differs from iteration 10's original subsample — sampling variance may also explain part of the gap.
- **Correctly parenting off iteration 10 did NOT guarantee matching its valset score**: Iteration 20 correctly used iteration 10 as parent (not a weak baseline) and got a reasonable valset (0.5680) — far better than the wrongly-parented iterations (0.44). But matching iteration 10's exact score requires exact reproduction of its code, not just architectural similarity.
- **Second aunt / complex derived family relation failure confirmed again**: Example 0 trace (gold='2', predicted='1') shows the agent attempts multi-hop family graph traversal via keyword search, fails to find the relation definition, then makes an off-by-one count error. The `search_wiki` tool cannot resolve "second aunt" as a semantic relationship — the agent must manually traverse the family graph: target → parents → grandparents → grandparents' siblings → filter female. This is 4–5 sequential lookups that the agent almost never completes correctly.
- **Two-pass architecture DOES NOT regress on single-answer questions**: The CompletenessCheck correctly gates Pass 2, protecting all single-answer questions from regression. This is a safe architectural pattern.
- **LLM-based ChainOfThought attribute extractors hallucinate**: In iteration 16, the attr_extractor added false-positive answers (e.g., "comptroller", "chief technology officer") not grounded in the retrieved passages. Do NOT use an LLM to extract specific attribute values from passages and add them to the answer set. Use deterministic extraction or require very tight evidence grounding.
- **Clarifying-question strings from ReAct leak into answers**: If ReAct returns a disambiguation question like "Which stone-collecting person do you mean?", downstream modules cannot recover this. The forward() method should detect question-like strings (starts with "Which"/"Who"/"What", ends with "?") in the answer list and replace them with `[]`.
- **Entity Frontier Tracker (iteration 18) produced a modest valset of 0.5591 — below best 0.6083 (−0.0492)**: The architecture replaced CompletenessCheck with EntityFrontierExtractor (ChainOfThought) to identify named intermediate entities and generate targeted reverse-link queries. Despite the architecture being logically sound, it was AGAIN parented off a weak candidate (candidate 8, subsample 5.137 — far below iter 10's 11.93), which limited its effectiveness. The subsample improvement was tiny (+0.131).
- **EntityFrontierExtractor correctly triggers multi-pass but ColBERT still fails on exact entity names**: The frontier extractor correctly identifies entities like "Derick Lafave" and suggests "children of Derick Lafave" as targeted queries, but ColBERT returns no results even for exact entity name queries. The bottleneck is retrieval, not query formulation. The architecture improvement cannot overcome this.
- **Replacing CompletenessCheck with EntityFrontierExtractor did not improve over CompletenessCheck baseline**: Iteration 10 (CompletenessCheck gating) achieved 0.6083; Iteration 18 (EntityFrontierExtractor) achieved 0.5591. The CompletenessCheck approach remains superior, though the parent mismatch confounds a clean comparison.
- **REPEATED PARENT MISTAKE (NOW THREE TIMES)**: Iterations 16, 18, and 19 all used weak parents (candidate 0, candidate 8, and candidate 0 again). EVERY new iteration MUST use iteration 10 (valset 0.6083) as parent. No exceptions. Parenting off candidate 0 (baseline subsample ~5-6) will ALWAYS produce a weak program (valset ~0.44) even if subsample improves — the subsample delta vs. a weak baseline is not meaningful.
- **Iteration 19 confirmed output verbosity is a major failure mode without `_normalize_answers`**: When parented off baseline (no deterministic normalizer), the react mutation produces answers like "Estella Beggs — hobby: leaves" instead of "leaves" and "Leonila Beggs — hobby: radio-controlled model collecting" instead of "radio-controlled model collecting". These score 0.0 F1. The `_normalize_answers` in iteration 10 is essential — it must be part of any descendant program. Any mutation parented off iteration 10 inherits this fix automatically.
- **Prompt mutation on react alone (without `_normalize_answers`) causes output formatting regression**: The subsample improved +0.925 (5.613→6.538), but the valset dropped to 0.44355. The subsample score is optimizing for a version of the problem without the normalizer, so the "improvement" is partially offset by format failures that the normalizer would have fixed.

