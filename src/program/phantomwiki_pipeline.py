import dspy
from src.program.counting_rm import CountingRM
from src.program.phantomwiki_module import PhantomWikiReAct

COLBERT_URL = "https://julianghadially--colbert-server-phantom-wiki-colbertserv-75bf93.modal.run/api/search"


# ---------------------------------------------------------------------------
# HopChainResolver — structured pre-pass for multi-hop chain decomposition
# ---------------------------------------------------------------------------

class HopChainExtractorSignature(dspy.Signature):
    """Decompose a multi-hop question into an ordered list of search queries, one per logical hop.
    For example, for 'Who is the mother of the cousin of the second uncle of Rosina Robey?',
    output: ['second uncle of Rosina Robey', 'cousin of {hop1}', 'mother of {hop2}'].
    Use placeholders like {hop1}, {hop2} to reference entities found in prior hops.
    Keep each hop query short and focused on a single relationship predicate."""

    question: str = dspy.InputField()
    hops: list[str] = dspy.OutputField(
        desc="ordered list of search queries, one per logical hop, with {hopN} placeholders for prior hop results"
    )


class EntityExtractorSignature(dspy.Signature):
    """Extract up to 4 entity names (people, places, or things) from the given passages that best
    match what the search query is looking for. Return only names, not descriptions."""

    query: str = dspy.InputField()
    passages: str = dspy.InputField()
    entities: list[str] = dspy.OutputField(
        desc="up to 4 entity names found in the passages that match the query predicate"
    )


class HopChainResolver(dspy.Module):
    """Lightweight pre-pass that decomposes a multi-hop question into sequential search queries,
    resolves each hop via dspy.Retrieve (no ReAct), and returns the final hop's entity candidates."""

    def __init__(self):
        self.hop_extractor = dspy.ChainOfThought(HopChainExtractorSignature)
        self.entity_extractor = dspy.ChainOfThought(EntityExtractorSignature)
        self.retrieve = dspy.Retrieve(k=10)

    def forward(self, question: str):
        # Step 1: Decompose the question into an ordered list of hop queries
        try:
            hop_result = self.hop_extractor(question=question)
            hops = hop_result.hops[:3]  # max 3 hops
        except Exception:
            return dspy.Prediction(chain_candidates=[])

        if not hops:
            return dspy.Prediction(chain_candidates=[])

        hop_entities: dict[int, list[str]] = {}  # hop index → resolved entity names

        for i, hop_query in enumerate(hops):
            # Substitute placeholders from previous hops (use first entity found)
            for j, entities in hop_entities.items():
                placeholder = f"{{hop{j + 1}}}"
                if placeholder in hop_query and entities:
                    hop_query = hop_query.replace(placeholder, entities[0])

            # Retrieve passages for this hop using dspy.Retrieve(k=10)
            try:
                retrieval = self.retrieve(hop_query)
                passages_text = "\n\n".join(retrieval.passages)
            except Exception:
                passages_text = ""

            # Extract up to 4 entity names from the retrieved passages
            try:
                extraction = self.entity_extractor(query=hop_query, passages=passages_text)
                entities = extraction.entities[:4]
            except Exception:
                entities = []

            hop_entities[i] = entities

        # The final hop's entity list becomes chain_candidates
        last_hop_idx = len(hops) - 1
        chain_candidates = hop_entities.get(last_hop_idx, [])

        return dspy.Prediction(chain_candidates=chain_candidates)


# ---------------------------------------------------------------------------
# FollowUpInvestigation — second-pass ReAct signature
# ---------------------------------------------------------------------------

class FollowUpInvestigation(dspy.Signature):
    """You are given a question and a list of answers already found via one investigation pass.
    Explore alternative relationship chains and paths NOT yet investigated to find additional answers.
    Treat already_found as a non-exhaustive partial result — there may be more valid answers
    reachable via different paths, relationships, or entity traversals that were not explored before."""

    question: str = dspy.InputField()
    already_found: list[str] = dspy.InputField(desc="answers discovered so far; treat as partial/non-exhaustive")
    answer: list[str] = dspy.OutputField(desc="additional answers found via unexplored paths")


# ---------------------------------------------------------------------------
# FinalAnswerSynthesizer — normalization post-processing step
# ---------------------------------------------------------------------------

class FinalAnswerSynthesizerSignature(dspy.Signature):
    """Synthesize a final, clean, deduplicated answer list from multiple candidate sources.
    Follow these rules strictly:
    (a) Merge all candidate sources (chain_candidates, pass1_answers, pass2_answers) into a unified pool.
    (b) For aggregation questions ('how many X does...'), extract ONLY unique numeric count values,
        stripping any 'Name: N' prefixed format to bare integers (e.g. 'Alice: 3' -> '3').
    (c) Remove any strings that say 'cannot be determined', 'not found', 'unknown', or similar
        error-like / null-result strings that do not constitute real answers.
    (d) Deduplicate semantically - treat name variants, casing differences, or equivalent values
        as one entry and keep only one representative form."""

    question: str = dspy.InputField()
    chain_candidates: list[str] = dspy.InputField(
        desc="entity candidates from hop-chain pre-pass"
    )
    pass1_answers: list[str] = dspy.InputField(
        desc="answers from first ReAct pass"
    )
    pass2_answers: list[str] = dspy.InputField(
        desc="answers from follow-up investigation pass"
    )
    answer: list[str] = dspy.OutputField(
        desc="final deduplicated merged answer list"
    )


# ---------------------------------------------------------------------------
# PhantomWikiReActPipeline — top-level pipeline
# ---------------------------------------------------------------------------

class PhantomWikiReActPipeline(dspy.Module):
    def __init__(self):
        self.rm = CountingRM(dspy.ColBERTv2(url=COLBERT_URL))
        self.program = PhantomWikiReAct()
        self.retrieve = dspy.Retrieve(k=7)
        self.hop_chain_resolver = HopChainResolver()
        self.followup_react = dspy.ReAct(
            signature=FollowUpInvestigation,
            tools=[self._search_wiki],
            max_iters=25,
        )
        self.final_synthesizer = dspy.ChainOfThought(FinalAnswerSynthesizerSignature)

    def _search_wiki(self, query: str) -> str:
        """Search the PhantomWiki corpus. Returns relevant passages."""
        results = self.retrieve(query)
        return "\n\n".join(results.passages)

    def forward(self, question):
        with dspy.context(rm=self.rm):
            # Pre-pass: HopChainResolver — lightweight multi-hop chain decomposition
            hop_result = self.hop_chain_resolver(question=question)
            chain_candidates = hop_result.chain_candidates

            # Pass 1: Primary ReAct agent
            result1 = self.program(question=question)

            # Pass 2: Follow-up investigation to explore unexplored paths
            result2 = self.followup_react(question=question, already_found=result1.answer)

            # Final synthesis: merge, normalize, and deduplicate all sources
            final = self.final_synthesizer(
                question=question,
                chain_candidates=chain_candidates,
                pass1_answers=result1.answer,
                pass2_answers=result2.answer,
            )

            return dspy.Prediction(answer=final.answer)
