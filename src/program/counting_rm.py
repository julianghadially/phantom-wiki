import sys
import threading
import time

import dspy
import requests
from requests.adapters import HTTPAdapter

# Shared, connection-pooled HTTP session for all ColBERT searches.
#
# Without this, each search did a bare ``requests.get(...)`` which opens a fresh
# TCP connection AND a fresh DNS lookup every call. Under concurrent load the
# local resolver (macOS mDNSResponder in particular) transiently fails to
# resolve the host, surfacing as ``NameResolutionError`` ([Errno 8]) and tanking
# eval scores. A pooled Session resolves the host once and reuses kept-alive
# connections across threads, so the per-search DNS/connection churn is gone.
#
# ``pool_maxsize`` must be >= the eval thread count so concurrent threads each
# get a reused connection rather than spilling over into new ones. 100 comfortably
# covers a 25-thread run.
_SESSION = requests.Session()
_adapter = HTTPAdapter(pool_connections=32, pool_maxsize=100)
_SESSION.mount("https://", _adapter)
_SESSION.mount("http://", _adapter)


class CountingRM(dspy.Retrieve):
    """Wraps any retrieval model to count queries, with timeout + backoff retry.

    The remote Modal ColBERT server for PhantomWiki serves a very large index and
    is prone to slow cold starts / transient timeouts under concurrent load. Three
    behaviors defend against that:

    1. ``timeout=240`` (4 min) instead of ColBERTv2's 10s default. The server keeps
       executing a request even after the client times out, so a short timeout that
       triggers a retry just piles a second request on top of the first and clogs
       the server. A long timeout lets a single request ride out a cold-start scale-up.
    2. Retries wait ``retry_backoff`` seconds (60s) between attempts. An instant
       retry re-hits the same exhausted/scaling server and re-resolves DNS,
       amplifying the failure; a pause lets it recover.
    3. All requests share a connection-pooled ``_SESSION`` (see above).

    Usage:
        rm = CountingRM(dspy.ColBERTv2(url=...))
        rm.reset_count()
        # ... run pipeline with dspy.context(rm=rm) ...
        num_retrievals = rm.call_count
    """

    def __init__(self, rm, timeout=240, max_retries=2, retry_backoff=60):
        super().__init__()
        self.rm = rm
        self.call_count = 0
        self._count_lock = threading.Lock()
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        # Override the default 10s timeout in ColBERTv2's underlying requests.
        self._patch_timeout()

    def _patch_timeout(self):
        """Monkey-patch the ColBERTv2 get request function to use our timeout
        and the shared connection-pooled session."""
        import dspy.dsp.colbertv2 as colbert_mod

        timeout = self.timeout

        def patched_get(url, query, k):
            payload = {"query": query, "k": k}
            res = _SESSION.get(url, params=payload, timeout=timeout)
            res.raise_for_status()
            res_json = res.json()
            if res_json.get("error"):
                raise ValueError(f"ColBERTv2 server returned an error: {res_json.get('message', 'Unknown error')}")
            if "topk" not in res_json:
                raise ValueError(f"ColBERTv2 server returned an unexpected response: {res_json}")
            topk = res_json["topk"][:k]
            topk = [{**d, "long_text": d["text"]} for d in topk]
            return topk[:k]

        colbert_mod.colbertv2_get_request_v2 = patched_get
        colbert_mod.colbertv2_get_request = patched_get

    def forward(self, query_or_queries, k=None, **kwargs):
        with self._count_lock:
            self.call_count += 1
        for attempt in range(self.max_retries + 1):
            try:
                return self.rm(query_or_queries, k=k, **kwargs)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt < self.max_retries:
                    print(
                        f"[WARNING] Retrieval timeout/error (attempt {attempt + 1}/"
                        f"{self.max_retries + 1}): {e}. Retrying in {self.retry_backoff}s...",
                        file=sys.stderr,
                    )
                    # Back off before retrying. The server keeps executing the
                    # original (timed-out) request and may still be scaling up;
                    # an instant retry just piles on and re-resolves DNS.
                    time.sleep(self.retry_backoff)
                else:
                    print(
                        f"[ERROR] Retrieval failed after {self.max_retries + 1} attempts: {e}",
                        file=sys.stderr,
                    )
                    raise

    def reset_count(self):
        with self._count_lock:
            self.call_count = 0
