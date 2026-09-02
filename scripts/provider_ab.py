"""A/B one provider against the other on a fixed PhantomWiki sample.

Runs the real ``PhantomWikiReActPipeline`` over the SAME rows with the SAME
thread count on one provider at a time, and records where the wall clock
actually goes: per row, per LM call, per retrieval call. The question this
answers is not "which provider scores higher" (same model, so scores should be
within noise) but "is DeepInfra's DeepSeek-V4-Flash deploy still slow, and does
it still produce hanging requests that stall the whole run".

Both arms MUST be run with the cross-provider fallback disabled
(``LM_FALLBACK=0``, which this script sets), otherwise a slow or failing
primary is silently served by the other provider and the timings describe
neither.

Run the arms SEQUENTIALLY, never concurrently: they share one remote ColBERT
server, and 50 in-flight retrievals would slow both arms in a way that has
nothing to do with the LM provider.

Both arms set ``LM_PROVIDER`` themselves, so they measure the named provider
whatever a normal run prefers. With no cover the circuit breaker in
``src/infra/lm_provider.py`` is inert and the full retry budget stands, so an
arm sees exactly the provider's own behaviour.

Usage:
    python scripts/provider_ab.py --provider deepseek   --n 200 --threads 25
    python scripts/provider_ab.py --provider deepinfra  --n 200 --threads 25
    python scripts/provider_ab.py --provider gmi        --n 200 --threads 25
    python scripts/provider_ab.py --compare output/provider_ab/gmi_*/summary.json \
                                            output/provider_ab/deepinfra_*/summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# Key files, most specific first. Loaded WITHOUT overwriting anything already
# exported, so the shell still wins -- these only fill in what it did not set.
ENV_FILES = (Path(".env.local"), Path(".env"))
DATA_PATH = Path("data/phantomwiki_trainval_omitsuperlong.json")


def load_env_files() -> None:
    for path in ENV_FILES:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))
OUT_ROOT = Path("output/provider_ab")
# A row is called out as a straggler when it takes this multiple of the median.
STRAGGLER_FACTOR = 3.0


# ---------------------------------------------------------------- timing ----

class Timers:
    """Per-row + global call timings, attributed by thread.

    DSPy runs one row entirely inside one worker thread, so a thread-local
    pointer to the row's record attributes every LM/retrieval call correctly.
    The global lists are the provider-latency measure that does not depend on
    that attribution holding.
    """

    def __init__(self):
        self._local = threading.local()
        self._lock = threading.Lock()
        self.lm_calls: list[float] = []
        self.lm_errors: list[str] = []
        self.rm_calls: list[float] = []
        self.inflight: dict[str, float] = {}

    def bind(self, record):
        self._local.record = record

    def _record(self):
        return getattr(self._local, "record", None)

    def track(self, kind: str, fn, *args, **kwargs):
        started = time.monotonic()
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            self._log(kind, time.monotonic() - started, exc)
            raise
        self._log(kind, time.monotonic() - started, None)
        return result

    def _log(self, kind: str, seconds: float, exc: BaseException | None) -> None:
        rec = self._record()
        with self._lock:
            (self.lm_calls if kind == "lm" else self.rm_calls).append(seconds)
            if exc is not None and kind == "lm":
                self.lm_errors.append(f"{type(exc).__name__}: {str(exc)[:200]}")
            if rec is not None:
                rec[f"{kind}_calls"] += 1
                rec[f"{kind}_seconds"] += seconds
                rec[f"max_{kind}_call_seconds"] = max(rec[f"max_{kind}_call_seconds"], seconds)
                if exc is not None:
                    rec["errors"].append(f"{kind}: {type(exc).__name__}")


def instrument(pipeline, timers: Timers):
    """Wrap the pipeline's LM and retrieval entry points with the timers."""
    lm, rm = pipeline.lm, pipeline.rm

    lm_forward, lm_aforward, rm_forward = lm.forward, lm.aforward, rm.forward

    def timed_lm_forward(*a, **kw):
        return timers.track("lm", lm_forward, *a, **kw)

    async def timed_lm_aforward(*a, **kw):
        started = time.monotonic()
        try:
            result = await lm_aforward(*a, **kw)
        except Exception as exc:
            timers._log("lm", time.monotonic() - started, exc)
            raise
        timers._log("lm", time.monotonic() - started, None)
        return result

    def timed_rm_forward(*a, **kw):
        return timers.track("rm", rm_forward, *a, **kw)

    lm.forward, lm.aforward, rm.forward = timed_lm_forward, timed_lm_aforward, timed_rm_forward


# ------------------------------------------------------------ statistics ----

def describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    def pct(p: float) -> float:
        return ordered[min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1))))]
    return {
        "n": len(ordered),
        "total": round(sum(ordered), 1),
        "mean": round(statistics.fmean(ordered), 2),
        "median": round(statistics.median(ordered), 2),
        "p90": round(pct(90), 2),
        "p99": round(pct(99), 2),
        "max": round(ordered[-1], 2),
    }


# ------------------------------------------------------------------ main ----

def load_sample(n: int, seed: int) -> list[dict]:
    """A deterministic sample -- the same seed gives both arms the same rows."""
    with DATA_PATH.open() as f:
        rows = json.load(f)
    if n >= len(rows):
        return rows
    return random.Random(seed).sample(rows, n)


def run(provider: str, n: int, threads: int, seed: int, watchdog: float) -> Path:
    load_env_files()
    os.environ["LM_PROVIDER"] = provider
    os.environ["LM_FALLBACK"] = "0"          # single-provider arm; see module docstring
    os.environ.setdefault("MLFLOW_TRACING", "0")  # tracing overhead would skew the timings

    # Imported after the env is set so the LM is built for the right provider.
    import dspy
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from src.metric.metric import phantomwiki_f1
    from src.program.phantomwiki_pipeline import PhantomWikiReActPipeline

    questions = load_sample(n, seed)
    pipeline = PhantomWikiReActPipeline()
    assert pipeline.lm.provider == provider, pipeline.lm.provider
    assert pipeline.lm._fallback is None, "fallback must be off for a single-provider arm"

    timers = Timers()
    instrument(pipeline, timers)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = OUT_ROOT / f"{provider}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{provider}] {len(questions)} rows x {threads} threads "
          f"-> {out_dir}  (model={pipeline.lm.model}, fallback=off)", flush=True)

    records: list[dict] = []
    print_lock = threading.Lock()
    t0 = time.monotonic()

    def process(q: dict) -> dict:
        rec = {
            "id": q["id"], "difficulty": q["difficulty"],
            "started_at": round(time.monotonic() - t0, 2),
            "lm_calls": 0, "lm_seconds": 0.0, "max_lm_call_seconds": 0.0,
            "rm_calls": 0, "rm_seconds": 0.0, "max_rm_call_seconds": 0.0,
            "errors": [],
        }
        timers.bind(rec)
        with timers._lock:
            timers.inflight[q["id"]] = time.monotonic()
        started = time.monotonic()
        try:
            result = pipeline(question=q["question"])
            rec["score"] = phantomwiki_f1(dspy.Example(answer=q["answer"]).with_inputs("question"), result)
        except Exception as exc:
            rec["score"] = 0.0
            rec["failed"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        finally:
            rec["seconds"] = round(time.monotonic() - started, 2)
            for key in ("lm_seconds", "rm_seconds", "max_lm_call_seconds", "max_rm_call_seconds"):
                rec[key] = round(rec[key], 2)
            with timers._lock:
                timers.inflight.pop(q["id"], None)
        with print_lock:
            done = len(records) + 1
            print(f"[{provider}] {done:>3}/{len(questions)}  {rec['seconds']:>7.1f}s  "
                  f"f1={rec['score']:.2f}  lm={rec['lm_calls']}call/{rec['lm_seconds']:.0f}s  "
                  f"rm={rec['rm_calls']}call/{rec['rm_seconds']:.0f}s  id={rec['id'][:40]}"
                  + (f"  FAILED {rec.get('failed', '')}" if rec.get("failed") else ""), flush=True)
        return rec

    # Watchdog: names the rows still in flight, so a hanging request is visible
    # while the run is happening rather than only in the post-hoc percentiles.
    stop = threading.Event()

    def watch():
        while not stop.wait(watchdog):
            now = time.monotonic()
            with timers._lock:
                slow = sorted(((now - s, i) for i, s in timers.inflight.items()), reverse=True)
            if slow:
                head = ", ".join(f"{i[:30]}@{age:.0f}s" for age, i in slow[:5])
                print(f"[{provider}] .. {len(slow)} in flight after "
                      f"{now - t0:.0f}s: {head}", file=sys.stderr, flush=True)

    threading.Thread(target=watch, daemon=True).start()

    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(process, q) for q in questions]
        for future in as_completed(futures):
            records.append(future.result())
    stop.set()
    wall = time.monotonic() - t0

    row_seconds = [r["seconds"] for r in records]
    median_row = statistics.median(row_seconds) if row_seconds else 0.0
    stragglers = sorted(
        (r for r in records if r["seconds"] > STRAGGLER_FACTOR * median_row),
        key=lambda r: -r["seconds"],
    )
    scores = [r["score"] for r in records]

    summary = {
        "provider": provider,
        "model": pipeline.lm.model,
        "fallback": "off",
        "rows": len(records),
        "threads": threads,
        "seed": seed,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_seconds": round(wall, 1),
        "mean_f1": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "failed_rows": sum(1 for r in records if r.get("failed")),
        "row_seconds": describe(row_seconds),
        "lm_call_seconds": describe(timers.lm_calls),
        "retrieval_call_seconds": describe(timers.rm_calls),
        "lm_calls_per_row": round(statistics.fmean([r["lm_calls"] for r in records]), 1) if records else 0,
        "lm_share_of_row_time": (
            round(sum(r["lm_seconds"] for r in records) / sum(row_seconds), 3) if sum(row_seconds) else 0.0
        ),
        "lm_errors": timers.lm_errors[:20],
        "num_lm_errors": len(timers.lm_errors),
        "straggler_threshold_seconds": round(STRAGGLER_FACTOR * median_row, 1),
        "stragglers": [
            {k: r[k] for k in ("id", "seconds", "lm_calls", "lm_seconds",
                               "max_lm_call_seconds", "rm_calls", "rm_seconds",
                               "max_rm_call_seconds", "difficulty", "score")}
            for r in stragglers[:15]
        ],
        "num_stragglers": len(stragglers),
        "tail_cost_seconds": round(wall - median_row, 1),
    }

    with (out_dir / "rows.jsonl").open("w") as f:
        for rec in sorted(records, key=lambda r: -r["seconds"]):
            f.write(json.dumps(rec) + "\n")
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + json.dumps(summary, indent=2), flush=True)
    return out_dir / "summary.json"


def compare(paths: list[str]) -> None:
    arms = [json.load(open(p)) for p in paths]
    keys = [
        ("wall_seconds", lambda a: a["wall_seconds"]),
        ("mean_f1", lambda a: a["mean_f1"]),
        ("rows", lambda a: a["rows"]),
        ("failed_rows", lambda a: a["failed_rows"]),
        ("row median s", lambda a: a["row_seconds"]["median"]),
        ("row p90 s", lambda a: a["row_seconds"]["p90"]),
        ("row max s", lambda a: a["row_seconds"]["max"]),
        ("LM call median s", lambda a: a["lm_call_seconds"]["median"]),
        ("LM call p99 s", lambda a: a["lm_call_seconds"]["p99"]),
        ("LM call max s", lambda a: a["lm_call_seconds"]["max"]),
        ("LM calls", lambda a: a["lm_call_seconds"]["n"]),
        ("stragglers", lambda a: a["num_stragglers"]),
        ("LM errors", lambda a: a["num_lm_errors"]),
    ]
    width = max(len(k) for k, _ in keys) + 2
    print("metric".ljust(width) + "".join(a["provider"].rjust(14) for a in arms))
    for label, get in keys:
        print(label.ljust(width) + "".join(f"{get(a):>14}" for a in arms))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("gmi", "deepinfra", "deepseek"))
    parser.add_argument("--n", type=int, default=200, help="rows to sample (default 200)")
    parser.add_argument("--threads", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0, help="same seed => same rows in both arms")
    parser.add_argument("--watchdog", type=float, default=120.0,
                        help="seconds between in-flight reports (0 disables)")
    parser.add_argument("--compare", nargs="+", metavar="SUMMARY.json")
    args = parser.parse_args()

    if args.compare:
        compare(args.compare)
        return
    if not args.provider:
        parser.error("--provider is required (or use --compare)")
    run(args.provider, args.n, args.threads, args.seed, args.watchdog or 1e9)


if __name__ == "__main__":
    main()
