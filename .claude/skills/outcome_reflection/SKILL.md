---
name: outcome_reflection
description: >
  Mid-session reflection on whether the iteration's change produced the
  expected improvement. Invoke this skill before exiting the iteration.
  Produces a written analysis and saves it to
  /memory/iteration_history_{iteration_idx}.md.
---

# Outcome reflection

You have just run one or more coding-agent changes inside the current
iteration and evaluated them on the training subsample. Before you exit this
session, reflect honestly on what happened. Future iterations will read your
reflection as part of `/memory/memory.md`, so the quality of your analysis
directly affects future experiments.

## What to write

Produce a short (200–600 word) analysis covering:

1. **Hypothesis recap** — one sentence. What were you testing?
2. **What changed** — the change request(s) you sent to the coding agent and,
   at a high level, what the coding agent actually did. Do not paste the full
   diff.
3. **Subsample progression** — the initial subsample score, the final
   subsample score, and whether the change improved, regressed, or was neutral.
   Reference specific `example_id`s that flipped or stayed the same.
4. **What the traces told you** — if you inspected traces, name the module(s)
   where behavior changed and how. If you did not inspect traces and should
   have, say so.
5. **Generalization assessment** — do you believe the change would help on
   the full validation set, or did it overfit to the subsample? Why?
6. **Follow-ups** — one or two concrete next experiments this iteration
   suggests, good or bad.

## Where to write it

Use the `Write` tool to save your analysis to:

```
/memory/iteration_history_{iteration_idx}.md
```

Use the current iteration index you were given at the start of the session.
Do not create files at any other path. Do not edit `/memory/memory.md` — the
valset reflection agent owns that file.

## Constraints

- Be honest. If the experiment failed, say so clearly and explain why you
  believe it failed.
- Do not propose additional coding-agent calls from inside this skill. Your
  reflection is the last step before exit.
- Do not reference specific private data or personally identifiable
  information from the dataset.
- The reflection is the file's entire content — no additional commentary
  outside the file.
