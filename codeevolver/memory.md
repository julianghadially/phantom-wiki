# CodeEvolver Memory
memory last updated at iteration number: 19

## Current Best Program
- **Iteration:** 12 (candidate 9)
- **Valset Score:** 0.7836676286676286
- **Previous Best (Iteration 11):** 0.7543716931216929
- **Baseline (Iteration 0):** 0.48995049139550406
- **Improvement over iter 11 best:** +0.0293 (~+3.9% relative gain)
- **Iteration 13 score:** 0.6640 — REGRESSION; best remains iter 12
- **Iteration 15 (current program) score:** 0.7784 — near-tie with best (−0.0052); best remains iter 12

## Observations

### Iteration 0 (Baseline)
- Initial baseline program.
- Score ~0.49 — slightly below 50% accuracy on valset.

### Iteration 1
- **Change:** Prompt mutation on `program.react.react` component (parent candidate 0).
- **Subsample score:** 13.09 (up from parent 12.70, +0.39).
- **Valset score:** 0.6022 — a large improvement over baseline (+0.112).
- The agent uses DSPy ReAct with ColBERTv2 retrieval over PhantomWiki corpus, up to 50 reasoning/action iterations per question.
- Strong performance on single-answer factual lookups and straightforward multi-hop relationship traversals.

### Iteration 2
- **Change:** Full code rewrite to three-stage Decompose→Enumerate→Synthesize pipeline (parent candidate 0).
- **Subsample score:** 0.0 (down from parent 10.65, −100%). **CATASTROPHIC FAILURE.**
- Root causes of failure:
  1. `ThreadPoolExecutor` in `ParallelAttributeFetcher` broke DSPy's retrieval context (thread-local state not propagated to workers), causing every example to crash.
  2. `attribute_to_collect` from Stage 1 was silently dropped and never passed to Stage 2.
  3. Hard entity cap of 4 would truncate multi-answer questions structurally.
  4. Count/scalar questions are architecturally incompatible with entity-enumerate→attribute-fetch pattern.
  5. Fallback to ReAct was effectively dead code (LLM almost always enumerates ≥1 entity).
- **Lesson: Major code rewrites are very high-risk. Stick to prompt mutations or small targeted code changes. If attempting a multi-stage pipeline, use sequential (not threaded) execution and ensure DSPy context propagates.**

### Iteration 3
- **Change:** Prompt mutation on `program.react.extract.predict` component (parent candidate 1).
- **Subsample score:** 11.022 (up from parent 10.422, +0.600).
- **Valset score:** 0.6469 — new best, improvement of +0.0447 over iteration 1.
- Remaining failure patterns (from subsample analysis):
  1. **Under-enumeration (primary):** Returns only 1 answer when multiple valid answers exist. Agent stops exploring after first match. Affects examples with multi-hop graph branches.
  2. **DOB lookup failures:** Questions involving a person born on very early dates (e.g., `0945-06-12`) fail to locate the person despite format variant attempts. May be a data indexing gap or year-padding edge case.
  3. **Incorrect "cannot be determined":** Agent gives up prematurely instead of attempting alternative lookup paths when a direct relation isn't listed.
  4. **Output format mismatch:** Returns "Person: count" strings instead of bare numeric counts for aggregation questions.
  5. **Aggregation vs. enumeration confusion:** Asked "how many friends does [granddaughter] have?", the agent summed across multiple granddaughters (6+1=7) instead of returning per-entity list [1, 6].
  6. **Wrong traversal:** Complex multi-hop chains (e.g., cousin of great-grandchild of great-grandfather) sometimes circle back to the starting entity.

### Iteration 4
- **Change:** Code change — Two-pass sequential ReAct with `FollowUpInvestigation` signature (parent candidate 2, iter 3 program).
- **Subsample score:** 12.632 (up from parent 10.517, +2.115 — ~20% relative gain). **New best subsample score.**
- **Valset score:** 0.6883 — new best, improvement of +0.0414 over iteration 3.
- Architecture: `PhantomWikiReActPipeline` now runs two passes:
  1. First pass: `PhantomWikiReAct` (max_iters=30, reduced from 50) — collects initial answers.
  2. Second pass: `FollowUpInvestigation` ReAct (max_iters=25) — explicitly told to explore unexplored paths, given `already_found` list.
  3. Outputs are merged deduplicated.
- What improved: Multi-entity enumeration questions (siblings, children, friends) benefited most. Second pass with `already_found` successfully found additional answers missed by first pass.
- Remaining failure patterns:
  1. **Non-existent seed data:** When anchor person/date doesn't exist in KB, both passes fail. Agent hallucinates "cannot determine" rather than returning empty list. Follow-up pass adds no value when `already_found` is empty/error.
  2. **Single-path early termination:** Agent pursues only one starting entity's path (e.g., one friend out of many) rather than exhausting all valid starting nodes.
  3. **Multi-occupant under-exploration:** "How many X does person with occupation Y have" requires enumerating ALL persons with that occupation. Agent finds only a fraction (e.g., 4 of 9, 1 of 7).
  4. **One-sided traversal (maternal-only):** For questions requiring paternal AND maternal branches, agent explores only one side (e.g., predicted 1 second aunt, gold was 2 — paternal grandparent siblings never explored).

### Iteration 5
- **Change:** Same two-pass architecture applied again from parent candidate 3 (iter 4 best). Subsample score: 12.621 (up from parent 11.583, +1.04). **Valset score: 0.5997 — REGRESSION from best (0.6883). Program is NOT the new best.**
- The regression (~−0.089 from iter 4 best) despite subsample improvement suggests overfit or sampling variance — the current valset sample may penalize this program's failure modes more heavily.
- Failure modes observed in subsample traces:
  1. **Output format mismatch on count-aggregation questions (critical, examples 0, 1, 14 — all score 0.0):** When question asks "how many X does each person with property Y have?", gold expects a list of unique count values (e.g., `['0','1','2','3',...]`), but model returns person-keyed pairs (e.g., `['Alice: 2', 'Bob: 4']`). The two-pass strategy accumulates MORE name-keyed pairs, making it worse. Trace example 0: gold was `['0','1','10','2','3','4','5','6','7','8','9']`, predicted `['Janey Hu: 1', 'Zelda Joubert: 4', ...]`. This is a systematic, high-frequency failure.
  2. **Retrieval dead-ends leading to premature "Cannot determine":** Agent explores one branch, fails, and gives up. Example: only tried Damon Moritz branch for nephews, never tried Adrian/Octavio Moritz branches. Also: 33 total steps wasted on DOB `0945-06-12` format variations without success.
  3. **Incomplete enumeration for shared attributes:** When multiple people share the same attribute (e.g., same DOB), search returns only one. Model stops after first hit, misses the rest (found 1 of 7 people sharing DOB 1050-09-16).
  4. **Second pass repeating failed strategies:** `already_found` context doesn't redirect follow-up agent to genuinely different paths; it re-runs near-identical queries. The "explore unexplored paths" instruction is not reliably followed when initial retrieval is empty.
- The two-pass approach helps enumeration tasks but **actively hurts** format-sensitive count-aggregation questions by producing more person-keyed pairs instead of deduplicating to unique numeric values.

## What Works / What Doesn't

### Works Well
- **Prompt mutations on ReAct components** consistently drive valset score improvements. Both `react.react` (iter 1) and `react.extract.predict` (iter 3) yielded gains.
- Multi-hop traversals with single valid paths are handled correctly.
- Single-answer factual and relational queries (e.g., "how many female cousins?") resolve accurately.
- Counting questions on straightforward single-parent sibling sets work well.


### Iteration 6
- **Change:** Planner → Multi-Branch Sequential Investigation → Synthesizer (parent candidate 0, iter 1 or baseline program).
- **Subsample score:** 10.317 (up from parent 8.530, +1.787). **Valset score: 0.6547 — still below best (0.6883) but better than iter 5 (0.5997).**
- Architecture: `QuestionDecomposer` (ChainOfThought) generates 2–4 investigation branches → `FocusedInvestigator` (ReAct, k=10, max_iters=15) runs sequentially per branch → `AnswerSynthesizer` (ChainOfThought) deduplicates/normalizes final answer. `PhantomWikiReAct` retained as fallback (only triggers if all branches return empty).
- What improved: The decomposer generates sensible multi-path branches, avoiding the threading DSPy-context issues of iter 2. Some improvement over parent on subsample.
- Remaining/new failure modes:
  1. **Error strings leaking through synthesizer:** Investigators return error messages like "No matching person...found" and the `AnswerSynthesizer` fails to strip them despite the prompt instruction, causing 0.0 scores.
  2. **Large multi-entity answer sets still under-recovered:** Branches with 7–17 expected answers return only 1–2 per branch; FocusedInvestigator terminates early after finding first satisfying path.
  3. **PhantomWiki-specific relational predicates misinterpreted:** "Second uncle," "second aunt," etc. are first-class graph predicates in PhantomWiki (not birth-order). Decomposer generates wrong branches (birth order searches, biographical searches) instead of direct graph lookups. Agents hallucinate plausible names instead of following the correct predicate.
  4. **Fallback logic gap:** Fallback to single-agent only triggers when `all_partial` is completely empty. When branches return partial/wrong/error answers, bad results pass to synthesizer unchecked.
  5. **Count-aggregation still partial:** Investigators miss many count categories because they stop after a few examples. Synthesizer collapses partials well, but completeness is limited.
  6. **Trace example 0 (second aunts):** Decomposer spent 3 searches on branch "identify Alan Denney node" (useful info already in first result), wasting iterations on meta-exploration. The real traversal (grandparent siblings) was never reached. Agent found 0,1,5 instead of gold 2.

## What Works / What Doesn't

### Works Well
- **Prompt mutations on ReAct components** consistently drive valset score improvements. Both `react.react` (iter 1) and `react.extract.predict` (iter 3) yielded gains.
- Multi-hop traversals with single valid paths are handled correctly.
- Single-answer factual and relational queries (e.g., "how many female cousins?") resolve accurately.
- Counting questions on straightforward single-parent sibling sets work well.
- **Sequential multi-branch investigation** (iter 6) shows improvement over single-pass but still doesn't reach the two-pass architecture's valset score.

### Doesn't Work / Failure Patterns
1. **Multi-answer incompleteness / under-enumeration:** The single most impactful failure — agent stops at first valid answer instead of exhausting all graph branches.
2. **Output format errors:** Prefixed answers ("Person: value") cause F1 failure; bare values required.
3. **Aggregation vs. per-entity answers:** Agent sums counts across entities when evaluator expects individual counts listed separately.
4. **DOB lookup for early-year dates:** Date format variants don't resolve certain persons.
5. **Ordinal relationship misinterpretation:** "Second uncle"/"second aunt" are direct graph predicates in PhantomWiki, NOT birth-order labels. Agents/decomposers that interpret them as "second by birth order" fail completely.
6. **Complex traversal errors:** Multi-hop chains with loops or branching can circle back to origin.
7. **Error string passthrough:** Synthesizer doesn't reliably strip error/failure messages from partial answers.
8. **Fallback only on empty results:** When branches return bad (non-empty) answers, fallback to reliable single-agent is bypassed.

## Major Opportunities
- **[CRITICAL] Fix output format for count-aggregation questions:** When asked "how many X does each person with property Y have?", gold expects a list of *unique count values* (e.g., `['0','1','2','3']`), NOT person-keyed pairs. A strong prompt directive and/or post-processing to strip "Name: " prefixes is still the highest-value fix.
- **[CRITICAL] Teach decomposer/agent about PhantomWiki-specific predicates:** "second_uncle", "second_aunt", "daughter_in_law", etc. are DIRECT GRAPH PREDICATES, not computed relationships. The decomposer/investigator must learn to do direct predicate lookups (search for "second uncle of [person]") rather than genealogical traversal. This could be addressed via prompt engineering with explicit examples.
- **Error string filtering before synthesizer:** Add a post-branch filter that drops any `all_partial` entry containing "cannot determine," "not found," "no matching," etc. before the synthesizer sees them.
- **Improved fallback triggering:** Consider triggering single-agent fallback when `len(all_partial) <= 1` or when answers are mostly error-like, not only when completely empty.
- **Multi-occupant / multi-starting-node enumeration:** Guide the agent to enumerate ALL starting nodes before answering (e.g., all persons with a given occupation).
- **Both-sides traversal enforcement:** For genealogy questions requiring paternal AND maternal branches (e.g., second aunts), agent systematically explores only one side.
- **Prompt mutations on new components:** `QuestionDecomposer`, `AnswerSynthesizer`, and `FocusedInvestigator` ReAct are all untested for prompt optimization.
- **Continue prompt mutations on iter 12 best:** The best program (iter 12, two-pass + AnswerNormalizer) is the strongest foundation. Prompt mutations on `FollowUpInvestigation` signature or `PhantomWikiReAct`'s `react.react` are lower-risk paths to further improvement. NOTE: AnswerNormalizer prompt mutations (iter 17) showed no effect — avoid re-trying them without a fundamentally different approach.
- **[NEW] PathDecomposer micro-investigation approach (iter 18) failed due to answer pollution from traversal artifacts.** If retried, must isolate "intermediate context" from "final answers" — only the `answer` field of micro-react should feed `already_found`, not raw reasoning trajectories or entity names encountered during traversal.
- **AnswerNormalizer has proven effective** (+0.029 valset gain, ~+18% subsample gain) for fixing "Name: count" format artifacts. The LLM-based ChainOfThought normalizer works; further prompt refinement of its rules could yield additional gains.
- **[HIGH VALUE] Fix AnswerNormalizer Rule 4:** When normalization removes all items (only error strings remain), Rule 4 forces the error string through unchanged — causing guaranteed 0.0 score. The fix: if all remaining items after Rules 1-3 are error strings and would be removed, return `[]` rather than original. This would recover points on DOB-failure and similar "no data found" cases.
- **[MEDIUM VALUE] Bidirectional relationship lookup:** When searching for spouse/relative and not found on Person A's page, also search for pages that list A as spouse/relative. This is a structural gap in the current search strategy.
- **Key remaining opportunity: retrieval for DOB-based anchor queries.** 6/20 subsample examples still score 0.0 because the ReAct agent cannot find a person given only a date-of-birth. Fuzzy date matching or alternative lookup strategies would unlock these zero-score cases.
- **Avoid full code rewrites** — iteration 2 showed these are extremely high-risk with DSPy context propagation pitfalls.

### Iteration 7
- **Change:** Prompt mutation on `program.react.extract.predict` component (parent candidate 0 — baseline single-pass ReAct).
- **Subsample score:** 10.183 (DOWN from parent 11.906, −1.722). **Valset score: not recorded (regression candidate, not evaluated as new best).**
- This was a **regression**: prompt change caused three failure modes:
  1. **Answer format contamination:** Model appended explanatory names in parentheses to numeric answers (e.g., `"2 (Judith Philips, Patti Fuchs)"` instead of `"2"`), causing F1 failures even when count was correct.
  2. **DOB-based anchor lookup regression:** Questions anchored on date-of-birth completely fail; model exhausts format variants then gives up. Retrieval search strategy was degraded.
  3. **Incomplete multi-answer enumeration:** "Stop-at-first-result" pattern became worse; multi-valued gold answers (7, 9, 14+ items) returned as single answers.
- **Lesson: Not all prompt mutations improve performance. Mutations on `react.extract.predict` can easily break output format discipline.**

### Iteration 8
- **Change:** Workspace-driven entity gap-filling architecture (parent candidate 2 — iter 4 two-pass best).
- **Subsample score:** 10.233 (up slightly from parent 9.833, +0.40).
- **Valset score: 0.5529 — MAJOR REGRESSION from best (0.6883, −0.1354). Current program.**
- Architecture: `GapAnalyzer` (ChainOfThought) analyzes initial ReAct trajectory to find un-resolved entities → `entity_react` (ReAct, max_iters=12) does targeted follow-up for up to 4 gap entities → all answers merged/deduped.
- Why valset regressed despite subsample improvement:
  1. **Answer bloat / format proliferation:** Each entity_react call re-discovers the same facts in different prose formats. Deduplication is exact-string only, so semantically identical answers with different wording all survive. F1 precision tanks.
  2. **Stale/hallucinated initial answers never discarded:** Initial pass may hallucinate wrong answers. Follow-up appends correct answers but wrong ones remain, hurting F1.
  3. **Wrong entity targeting:** GapAnalyzer latches onto ancestors/intermediaries rather than true target entities. Entity-react investigates wrong people.
  4. **`investigation_complete` never fires:** System always runs all 4 entity follow-ups even when not needed, amplifying bloat and noise.
  5. **Added complexity without clear win:** The gap-filling loop adds cost and noise for many question types where the simple two-pass approach from iter 4 was already competitive.
- **Lesson: Answer deduplication must be semantic, not exact-string. Adding more LLM passes without controlling output format creates precision regression.**

### Iteration 9
- **Change:** Two-pass sequential ReAct with `FollowUpInvestigation` (same architecture as iter 4, but parent candidate 3 — the iter 4 best — with program.react.extract.predict as selected component).
- **Subsample score:** 12.734 (up from parent 11.520, +1.214, ~10.5% relative). **Valset score: 0.6848 — just below best (0.6883, −0.0036). Current program.**
- This re-implemented the exact same two-pass strategy as iter 4. The slight valset gap vs iter 4's 0.6883 is within noise/sampling variance.
- 10 of 20 subsample examples scored perfect 1.0.
- Remaining failure modes (consistent with prior iterations):
  1. **Entity lookup failure cascading through both passes:** When first pass can't locate an entity (e.g., by DOB), it returns "Cannot determine." Second pass inherits wrong framing and also fails. Two-pass doesn't help when root cause is failed retrieval.
  2. **Answer formatting inconsistency:** Agent returns labeled answers like `"Refugio Crum: 1"` instead of bare `"1"`, scoring 0.0 even when data was correctly retrieved. Prompt/output format issue not addressed by two-pass.
  3. **Persistent under-enumeration on large answer sets:** For questions with 6–9 valid answers, both passes still miss majority (e.g., 1/6, 4/9, 2/6 found). 25-iteration budget on follow-up insufficient for exhaustive enumeration.
  4. **Wrong path selection:** Agent follows incorrect relationship chain in both passes, leading to wrong attribute (e.g., "microbiology" instead of "antiquities").

## Other Important Notes
- **Current best:** Iteration 12, valset score 0.7836 (NEW BEST). Previous best was iter 11 at 0.7543. Current program = iter 12 at 0.7836.
- Architecture (iter 4): Two-pass DSPy ReAct pipeline. Pass 1: `PhantomWikiReAct` (max_iters=30). Pass 2: `FollowUpInvestigation` ReAct (max_iters=25) with `already_found` context. Both backed by ColBERTv2 retrieval (k=7) over PhantomWiki.
- Architecture (iter 6): Planner→Multi-Branch→Synthesizer. `QuestionDecomposer` (CoT) → `FocusedInvestigator` (ReAct, k=10, max_iters=15) × N branches → `AnswerSynthesizer` (CoT). Fallback to `PhantomWikiReAct` only if all branches empty.
- Metric: token-level set F1 between predicted and gold answer lists (via `phantomwiki_f1_feedback`).
- Validation sample changes between iterations — avoid overfitting to any single sample.
- Maintain diversity of candidate programs across iterations to explore the solution space broadly.
- The question types include: (a) aggregation/count queries ("how many X does each person with occupation Y have?" — expects unique count values as list), (b) multi-hop traversals of varying depth/complexity, (c) single-entity attribute lookups. Count-aggregation questions with expected unique-value lists are currently the most broken.
- `search_wiki` tool does NOT accept a `top_k` parameter — agent self-corrects but the prompt should avoid suggesting `top_k`.
- PhantomWiki has specific relational predicates (second_uncle, second_aunt, daughter_in_law, etc.) that must be queried directly — NOT computed via genealogical traversal. This is a major insight for decomposer/agent prompt engineering.
- The multi-branch architecture is conceptually promising but the quality of the final answer depends heavily on: (1) whether the decomposer generates correct branch types, (2) whether individual branches exhaust all matching nodes, (3) whether the synthesizer correctly normalizes formats.
- **Iterations 5, 6, 7, 8, and 10 all fell below iter 4's best valset score (0.6883)**. Iter 11 achieved a major jump to 0.7543 — the new best. The gap-filling approach (iter 8, 10) and single-pass prompt mutations (iter 7) both regressed significantly.
- **Iteration 9 re-implemented the exact same two-pass architecture from iter 4 and scored 0.6848 (−0.0036 from best)**. The near-tie validates that the iter 4 two-pass approach is robustly the best architecture found. The tiny gap may be sampling variance.
- **Gap-filling architectures with multiple LLM passes produce answer bloat unless output normalization is applied.** Any additional ReAct passes must output ONLY atomic answer values (names, counts, dates) — not prose descriptions or entity-keyed pairs.
- **The iter 12 architecture (two-pass ReAct + AnswerNormalizer) is the current strongest foundation.** The AnswerNormalizer successfully fixed "Name: count" format issues and achieved the new best valset score 0.7836. Best next steps are: (1) prompt mutations on `FollowUpInvestigation` or `PhantomWikiReAct` signatures, (2) improving retrieval for DOB-based queries, (3) better enumeration of all unique count values for aggregation questions.
- **CRITICAL: The AnswerNormalizer is NON-OPTIONAL.** Iteration 13 proved that removing the AnswerNormalizer from the two-pass pipeline causes a regression from 0.7836 to 0.6640 (−0.1196). ALL future iterations MUST use iter 12 (candidate 9) as the parent, not any earlier candidate. Starting from iter 8 or other pre-normalizer candidates discards the hard-won format fix.
- **Avoid: adding more LLM passes without strict output format control; exact-string deduplication when answers can be semantically equivalent but textually different; GapAnalyzer-style meta-analysis that can target wrong entities; using pre-iter-12 candidates as parents (they lack the AnswerNormalizer).**
- **⚠️ CRITICAL RECURRING ERROR — Iterations 13, 16, AND 19 all used the wrong parent (candidate 0 / baseline), all resulting in major valset regressions. This error has now occurred THREE times. The reflection agent MUST verify the parent is candidate 9 (iter 12) BEFORE applying any change. ⚠️**
- **Iteration 16 confirmed (AGAIN): Using the wrong parent (candidate 0 / baseline) causes a catastrophic regression even when the new architecture is conceptually sound.** The 4-pass completeness+synthesis approach (iter 16) scored only 0.5388 vs best 0.7836 — a −0.2448 drop — primarily because it discarded the iter 12 two-pass ReAct + AnswerNormalizer foundation. **ALL future iterations MUST use iter 12 (candidate 9) or a descendant as the parent.**
- **The `AnswerCompletenessChecker` + `AnswerSynthesizer` idea from iter 16 has merit** but must be layered ON TOP of iter 12's two-pass architecture, not as a replacement. The synthesizer successfully handles format normalization. The completeness checker generates good queries but `self.rm` single-retrievals are insufficient for exhaustive enumeration — a short second ReAct pass would be more effective for aggregation questions where the checker flags incompleteness.

### Iteration 10
- **Change:** Code change — Added `HopChainResolver` pre-pass + `FinalAnswerSynthesizer` post-processing to two-pass ReAct pipeline (parent candidate 7).
- **Subsample score:** 10.744 (DOWN from parent 11.886, −1.14). **Valset score: not recorded (regression candidate).**
- Architecture: `HopChainResolver` decomposes question into ordered hop queries and resolves via pure retrieval (no ReAct), producing `chain_candidates`. `FinalAnswerSynthesizer` (CoT) merges all three result sets (chain_candidates, pass1, pass2), strips error strings, normalizes aggregation counts, deduplicates semantically.
- Why it regressed:
  1. **Empty-hop hallucination:** When intermediate hop returns empty, unfilled `{hop2}` placeholder is passed as literal string to retriever, surfacing unrelated passages. Entity extractor fabricates names (e.g., "Raina Hoppe").
  2. **FinalAnswerSynthesizer over-trusts chain candidates:** When ReAct passes fail, synthesizer falls back to broken chain candidates without validation.
  3. **Aggregation format mismatch:** Synthesizer's numeric extraction can't parse "name: count" compound strings from ReAct, returns `[]`.
  4. **DOB-based retrieval failure:** Semantic retrieval can't find exact date matches; chain collapses completely.
- **Lesson: Structured pre-pass chains are brittle when intermediate hops return empty — must short-circuit to ReAct when hops fail. FinalAnswerSynthesizer needs input validity gating.**

### Iteration 11
- **Change:** Code change — Two-pass sequential ReAct with `FollowUpInvestigation` (same core architecture as iter 4 and 9, but started from parent candidate 3 with `followup_react.react` as selected component). Reduced `max_iters` from 50→30 in `phantomwiki_module.py`.
- **Subsample score:** 16.439 (up from parent 15.052, +1.387, ~9.2% relative). **Valset score: 0.7543 — NEW BEST at that time, improvement of +0.0660 over previous best (iter 4: 0.6883).**
- Architecture: Same two-pass DSPy ReAct pipeline as iter 4/9. Pass 1: `PhantomWikiReAct` (max_iters=30). Pass 2: `FollowUpInvestigation` ReAct (max_iters=25) with `already_found` context. Answers merged deduplicated.
- What improved: Substantial jump to 0.7543 from prior best 0.6883. The two-pass strategy is clearly the strongest architecture found.
- Remaining failure modes: "Name: count" format artifacts, "Cannot be determined" error strings leaking into answers, incomplete enumeration on large answer sets, retrieval failure on early-year DOB lookups.

### Iteration 12
- **Change:** Code change — Added `AnswerNormalizerSignature` + `dspy.ChainOfThought` post-processing module to two-pass ReAct pipeline (parent candidate 8). The normalizer strips "Name: count" format artifacts by extracting unique numeric counts, removes error strings ("Cannot be determined", "not found", etc.), and falls back to original answers if filtering would yield an empty list.
- **Subsample score:** 11.122 (up from parent 9.430, +1.69, ~18% relative gain). **Valset score: 0.7836 — NEW BEST, improvement of +0.0293 over iter 11 (0.7543).**
- Architecture: Same two-pass DSPy ReAct pipeline as iter 11, plus `AnswerNormalizer` post-processing step after answers are merged/deduped.
- What improved: "Name: count" format failures (examples 2/3 in subsample) now score >0.6 instead of 0.0. The normalizer correctly extracts numeric count values from person-keyed pair strings.
- Remaining failure modes:
  1. **Retrieval failure on entity not found (DOB-based lookups):** 6 examples still return "Cannot be determined" after both ReAct passes fail. The normalizer's Rule 4 (preserve original if filtering would empty list) returns the error string unchanged. Root cause is retrieval, not formatting.
  2. **Incomplete enumeration of all unique count values:** Even after format normalization, count-aggregation questions only recover ~5–6 of 11–12 expected unique values. ReAct passes don't exhaustively enumerate all possible count values.
  3. **Wrong entity disambiguation:** Examples 10/15 retrieve the wrong person; unrelated to formatting. Entity disambiguation in ReAct reasoning needs improvement.
  4. **AnswerNormalizerSignature traces not always visible:** Normalizer may silently fall back to original without logging; makes debugging difficult.

### Iteration 13
- **Change:** Code change — Reimplemented two-pass sequential ReAct with FollowUpInvestigation (parent candidate 8). Critically, this version does NOT include the `AnswerNormalizer` post-processing that was key to iter 12's success (0.7836). The change was applied to the `followup_react.extract.predict` component.
- **Subsample score:** 11.482 (up from parent 11.149, +0.333, ~3% relative). **Valset score: 0.6640 — MAJOR REGRESSION from best (0.7836, −0.1196).**
- The regression places iter 13 well below iter 12 and even below iter 9 (0.6848). The score is also below iter 4 (0.6883).
- **Root cause of regression:** This iteration built the two-pass pipeline from parent candidate 8 (gap-filling architecture, iter 8) but did NOT include the `AnswerNormalizerSignature` + `dspy.ChainOfThought` post-processing module that was the defining feature of iter 12. Without the normalizer, output format pollution returns (person-keyed "Name: count" strings pass through unfiltered), recreating the failure mode that iter 12 specifically fixed.
- Confirmed failure modes from subsample:
  1. **Output format pollution (examples 11, 15, 17 — scored 0.0):** Agent returns `"Carmine Libby — waste management officer"` instead of `"waste management officer"`, `"Janey Hu: 1"` instead of `"1"`. Without the AnswerNormalizer, these pass through uncorrected.
  2. **Premature "Cannot determine" (examples 0, 4, 10, 12 — scored 0.0):** First pass declares data absent; second pass inherits wrong framing and also fails. Deep traversal (great-uncles, DOB-based) still broken.
  3. **Under-enumeration persists:** Example 6 found 4 of 9 correct answers; example 1 found 1 of 4. Two-pass doesn't reliably broaden coverage for multi-sibling/multi-child enumeration.
- **Lesson: The AnswerNormalizer is NON-OPTIONAL on top of the two-pass architecture. Any future iteration must start from iter 12's program (two-pass + AnswerNormalizer) as the base, not from iter 8 or earlier candidates without the normalizer.**

### Iteration 15
- **Change:** Code change — Added `AnswerNormalizerSignature` + `dspy.ChainOfThought` post-processing module to two-pass ReAct pipeline (parent candidate 9 — iter 12 best).
- **Subsample score:** 14.161 (up from parent 13.780, +0.381, ~2.8% relative). **Valset score: 0.7784 — slightly below best (0.7836, −0.0052). Near-tie with best.**
- Architecture: Same as iter 12 — two-pass DSPy ReAct + AnswerNormalizer. The change was applied to the `followup_react.extract.predict` component.
- What improved: "Name: count" format artifacts now correctly extracted to plain numeric strings (examples 2/3 score 0.62 and 1.0, up from 0.0). Clean pass-through for well-formed answers confirmed correct.
- **Systematic failure mode of Rule 4 confirmed:** When ReAct agent fails entirely and returns only a single verbose error string, Rule 4 ("never return empty") forces the error string through unchanged to the final answer (guaranteeing score 0.0). Examples 9 and 12 are affected. A better fallback would be to return `[]` (empty list) rather than propagating the error string.
- Remaining failure modes:
  1. **DOB-based retrieval failure (examples 9, 12):** Agent exhausts format variants for early-year dates without finding the person; Rule 4 prevents empty-list fallback.
  2. **Bidirectional spouse lookup gap (example 12):** When Person A's page doesn't list spouse, but spouse's page lists A — agent misses the relationship.
  3. **Multi-path ambiguity (examples 1, 5, 13, 15, 19, 20):** Agent finds plausible but incomplete/incorrect set when many relationship chains coexist.
  4. **Second aunt misidentification (example 15):** Agent reads the wrong parent's sibling list when tracing "second aunt."
- **Opportunities identified:**
  1. Fix Rule 4: when only error strings remain after normalization, return `[]` rather than the error string — this alone would recover points on DOB-failure cases.
  2. Bidirectional spouse/relative lookup: if Person A's page lacks spouse field, search for pages listing A as spouse.
  3. Tighter entity resolution for second-aunt/uncle traversal.

### Iteration 16
- **Change:** Code change — 4-pass "Answer Completeness + Targeted Search + Synthesis" pipeline. **Parent candidate 0 (baseline single-pass ReAct — NOT iter 12 best).** Added `AnswerCompletenessChecker` (ChainOfThought, detects incompleteness, generates follow-up queries) and `AnswerSynthesizer` (ChainOfThought, merges initial + supplemental results into deduplicated clean answer list). ReAct module reverted to max_iters=50, k=7.
- **Subsample score:** 13.1618 (up from parent 13.0079, +0.1538, ~1.2% relative). **Valset score: 0.5388 — MAJOR REGRESSION from best (0.7836, −0.2448). Well below iter 12.**
- Architecture: (1) Initial ReAct investigation → (2) `AnswerCompletenessChecker` generates follow-up queries → (3) direct `self.rm` searches for each follow-up query → (4) `AnswerSynthesizer` merges all findings.
- Why valset regressed severely:
  1. **Wrong parent (critical):** Started from candidate 0 (baseline), not iter 12. The AnswerNormalizer and two-pass FollowUpInvestigation from iter 12 were discarded. The synthesis step (AnswerSynthesizer) covers some of AnswerNormalizer's format normalization, but the entire two-pass ReAct architecture is gone.
  2. **Aggregation enumeration still broken:** For "how many X does EACH person with property Y have?" questions, the initial ReAct finds one entity and stops. The completeness checker generates sensible meta-queries, but `self.rm` single retrievals return raw passage text, not an exhaustive entity list. Synthesizer biases toward the initial answer and ignores supplemental text, yielding one-element answers instead of full count distributions.
  3. **Multi-branch relationship under-enumeration:** The follow-up queries re-ask the same high-level question rather than targeting each unexplored branch explicitly.
  4. **Synthesizer strong prior toward initial answers:** When supplemental passages don't explicitly enumerate additional answers, the synthesizer ignores them — defeating the purpose of the follow-up pass.
  5. **DOB-based retrieval still broken:** When initial ReAct fails to find person by DOB, completeness checker generates date-format variants that also fail retrieval; synthesizer returns empty list.
- What worked in this approach:
  - `AnswerSynthesizer` successfully normalizes format (strips "Cannot be determined" strings, extracts plain values) for single-answer questions.
  - `AnswerCompletenessChecker` correctly identifies when answers are incomplete and generates on-target follow-up queries in most cases.
  - Simple 1-answer questions handled cleanly (completeness checker returns empty follow-up list, avoiding wasted calls).
- **Lesson: This 4-pass approach is conceptually sound but loses the critical two-pass ReAct foundation and AnswerNormalizer from iter 12. The `self.rm` single-retrieval follow-ups cannot replace a full ReAct pass for enumeration. This architecture needs to be built ON TOP of iter 12 (two-pass + AnswerNormalizer), not as a replacement.**

### Iteration 17
- **Change:** Prompt mutation on `answer_normalizer.predict` component (parent candidate 11 — iter 12 best two-pass + AnswerNormalizer).
- **Subsample score:** 12.8 (same as parent 12.8, delta: +0.0). **No valset score recorded (neutral / no improvement).**
- Architecture: Identical to iter 12 — two-pass DSPy ReAct + AnswerNormalizer. Only the normalizer prompt was mutated.
- What happened: The normalizer was completely benign — correctly-passing examples stayed correct, failing examples stayed failing. The ChainOfThought normalizer still failed to remove single-item error strings ("Cannot be determined — no person with DOB...") because Rule 4 ("return original if filtering would empty list") preserves them.
- Confirmed: The AnswerNormalizer prompt mutation had no effect because the underlying failures (DOB retrieval failure, incomplete enumeration) are upstream reasoning issues — not formatting issues. No normalization prompt can recover answers that were never retrieved.
- **Lesson: AnswerNormalizer prompt mutations have reached diminishing returns. The persistent failures are retrieval-level (DOB lookups, under-enumeration), not format-level.**

### Iteration 18
- **Change:** Code change — Added PathDecomposer + Sequential Micro-Investigation pass (Pass 1.5) between existing Pass 1 and Pass 2 (parent candidate 9 — iter 12 best). Three-pass architecture: Pass 1 (PhantomWikiReAct, 30 iters) → Pass 1.5 (PathDecomposer ChainOfThought + up to 4 TargetedInvestigation ReAct, 8 iters each) → Pass 2 (FollowUpInvestigation, 25 iters) → AnswerNormalizer.
- **Subsample score:** 7.743 (DOWN from parent 10.967, −3.224, ~29% relative decline). **No valset score recorded (major regression). NOT a new best.**
- Architecture: PathDecomposer identifies up to 4 unexplored entity/relationship paths from Pass 1's partial answers. TargetedInvestigation micro-react (max_iters=8) focused per path. Micro answers folded into `already_found` for Pass 2.
- Why it regressed severely:
  1. **Answer pollution from intermediate traversal artifacts:** Micro-react returned entity names discovered EN ROUTE (graph traversal artifacts like intermediate ancestors, not final answers), which were folded into `already_found` and survived AnswerNormalizer into final output. Tanked precision/F1.
  2. **Incomplete branch coverage:** With only 8 iters per targeted path, micro-react explored only one branch of multi-branch family trees, missing the majority of expected answers.
  3. **Generational confusion in synthesis:** `already_found` aggregation misidentified generational levels; wrong generational anchors caused hallucinated answers.
  4. **Over-specificity traps:** Anchoring micro-react to a specific path meant a single wrong entry point caused the entire sub-investigation to fail silently. The old FollowUpInvestigation explored more freely and was more resilient.
  5. **Convergence on wrong entity clusters:** Examples with same question structure both returned the same wrong entity list; Pass 2 couldn't override once wrong context was baked into `already_found`.
- **Lesson: Micro-react intermediate passes generate traversal artifact pollution. Intermediate results must be labeled "context" vs "answers" — only final answer fields should enter `already_found`. If PathDecomposer revisited, must strip intermediate traversal artifacts before merging with answers.**

### Iteration 19 (Current)
- **Change:** Prompt mutation on `program.react.react` component (parent candidate 0 — BASELINE single-pass ReAct). **WRONG PARENT — discarded iter 12's two-pass + AnswerNormalizer foundation.**
- **Subsample score:** 10.300 (up from parent 9.787, +0.513). **Valset score: 0.5968 — MAJOR REGRESSION from best (0.7836, −0.1868).**
- Architecture: Single-pass DSPy ReAct only (max_iters=50). No FollowUpInvestigation, no AnswerNormalizer. This is essentially the baseline architecture from iteration 0/1 with a slightly mutated prompt.
- Why valset regressed: The parent was candidate 0 (baseline) — discarding the two-pass ReAct and AnswerNormalizer that are the core of iter 12's superior performance. Without AnswerNormalizer, "Name: count" format artifacts return. Without FollowUpInvestigation, multi-answer enumeration is incomplete.
- Confirmed failure modes from subsample: DOB lookup failures (format mismatch), incomplete multi-hop family traversal (returns 1 of N expected), answer verbosity (count + names in parentheses causes F1 fail), per-person vs unique-values format confusion.
- **This is the THIRD time the wrong parent (candidate 0) was used (also iterations 13 and 16). This error pattern must be actively prevented.**

## Score History Summary
| Iteration | Valset Score | Notes |
|-----------|-------------|-------|
| 0 | 0.4900 | Baseline |
| 1 | 0.6022 | Prompt mutation on react.react |
| 2 | ~0.0 | Catastrophic failure (ThreadPoolExecutor broke DSPy context) |
| 3 | 0.6469 | Prompt mutation on react.extract.predict |
| 4 | 0.6883 | Two-pass ReAct with FollowUpInvestigation (prev best) |
| 5 | 0.5997 | Regression despite subsample gain |
| 6 | 0.6547 | Multi-branch planner architecture |
| 7 | N/A | Regression (answer format contamination) |
| 8 | 0.5529 | Major regression (answer bloat from gap-filling) |
| 9 | 0.6848 | Two-pass re-implementation (near-tie with iter 4) |
| 10 | N/A | Regression (HopChainResolver + FinalAnswerSynthesizer) |
| 11 | 0.7543 | NEW BEST at time — Two-pass ReAct (same arch, max_iters 50→30) |
| 12 | **0.7836** | **CURRENT BEST** — Two-pass ReAct + AnswerNormalizer post-processing |
| 13 | 0.6640 | Regression — Two-pass without AnswerNormalizer (wrong parent: candidate 8) |
| 14 | N/A | (no history file reviewed) |
| 15 | 0.7784 | Near-tie with best — Two-pass + AnswerNormalizer (from iter 12 parent) |
| 16 | 0.5388 | MAJOR REGRESSION — 4-pass completeness+synthesis (wrong parent: candidate 0) |
| 17 | N/A | Neutral — AnswerNormalizer prompt mutation, no change (parent: iter 12 best) |
| 18 | N/A | MAJOR REGRESSION on subsample (−29%) — PathDecomposer+Micro-Investigation (parent: iter 12), answer pollution |
| 19 | 0.5968 | MAJOR REGRESSION — Single-pass ReAct prompt mutation (wrong parent: candidate 0 again) |

