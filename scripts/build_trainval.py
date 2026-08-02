"""Build the combined train+val trainset for the PhantomWiki VANILLA baseline arm.

Writes:

    data/phantomwiki_trainval_omitsuperlong.json    (228 rows = train 113 + val 115)

Why this exists: per ``specs/research_plan.md`` in CodeEvolver, the controlled
quantity across arms is the *total* data available to train on = train + val.
The CodeEvolver arms (asa / greedy / pareto) spend it as 113 train + a 115-row
val used for admission decisions. The vanilla loop makes no validation-split
decision and spends zero eval budget on a valset, so it receives the same rows
as one combined pool. Without this file vanilla would train on 113 rows against
the CE arms' 228 and the comparison would be invalid.

The engine will not do this merge: ``DatasetConfig`` keeps splits in separate
files by design so an iteration can never read valset rows as training signal.
Building the combined pool is the caller's job.

This file contains GOLD ANSWERS (``answer`` / ``solution_traces`` / ``prolog``)
and must be listed in ``additional_deny_paths`` for *every* arm, not just
vanilla -- it carries the full valset, so leaving it readable hands the CE arms
their own holdout. ``test`` is untouched and stays held out for all arms.

The ``_omitsuperlong`` variants are the ones the experiment configs use; the
plain ``phantomwiki_{train,val}.json`` files are not part of this merge.

Usage:
    python scripts/build_trainval.py            # build if missing
    python scripts/build_trainval.py --force    # rebuild
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DATA_DIR = Path("data")
PARTS = ("train", "val")
OUT_PATH = DATA_DIR / "phantomwiki_trainval_omitsuperlong.json"
HOLDOUT_PATH = DATA_DIR / "phantomwiki_test_omitsuperlong.json"
ID_KEY = "id"


def build(force: bool = False) -> None:
    if OUT_PATH.exists() and not force:
        existing = json.load(open(OUT_PATH))
        print(f"SKIP existing {OUT_PATH} ({len(existing)} rows); use --force")
        return

    combined: list = []
    for name in PARTS:
        path = DATA_DIR / f"phantomwiki_{name}_omitsuperlong.json"
        with open(path) as f:
            rows = json.load(f)
        print(f"  {path} -> {len(rows)} rows")
        combined.extend(rows)

    # Hard guarantee: the held-out test set must not leak into the train pool.
    with open(HOLDOUT_PATH) as f:
        holdout_ids = {r[ID_KEY] for r in json.load(f)}
    leaked = holdout_ids & {r[ID_KEY] for r in combined}
    assert not leaked, f"{len(leaked)} test rows leaked into trainval!"

    unique = len({r[ID_KEY] for r in combined})
    if unique != len(combined):
        print(f"  note: {len(combined) - unique} duplicate {ID_KEY}(s) across train/val (preserved)")

    DATA_DIR.mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Wrote {OUT_PATH} ({len(combined)} rows, {unique} unique {ID_KEY}s, 0 test overlap)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the PhantomWiki vanilla train+val pool")
    parser.add_argument("--force", action="store_true", help="Overwrite the existing file")
    build(force=parser.parse_args().force)


if __name__ == "__main__":
    main()
