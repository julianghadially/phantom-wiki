"""Fixed infrastructure: everything here is OUTSIDE the program under optimization.

CodeEvolver evolves ``src/program`` (its config names
``src.program.phantomwiki_pipeline.PhantomWikiReActPipeline`` as the parent
module). This package holds the wiring that must survive that untouched -- the
task-LM provider routing and the tracing bridge. The benchmark pins the model,
so provider and model ids live here and nowhere else; see
``docs/provider_fallback.md`` (contract item C8).

Do not modify anything in this package as part of optimizing the program.
"""
