import sys
import time

import dspy
import requests


class CountingRM(dspy.Retrieve):
    def __init__(self, rm, timeout=30, max_retries=1):
        super().__init__()
        self.rm = rm
        self.call_count = 0
        self.timeout = timeout
        self.max_retries = max_retries
        # Override the default 10s timeout in ColBERTv2's underlying requests
        self._patch_timeout()

    def _patch_timeout(self):
        """Monkey-patch the ColBERTv2 get/post request functions to use our timeout."""
        import dspy.dsp.colbertv2 as colbert_mod

        orig_get = colbert_mod.colbertv2_get_request_v2.__wrapped__
        orig_post = colbert_mod.colbertv2_post_request_v2.__wrapped__
        timeout = self.timeout

        def patched_get(url, query, k):
            payload = {"query": query, "k": k}
            res = requests.get(url, params=payload, timeout=timeout)
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
        self.call_count += 1
        for attempt in range(self.max_retries + 1):
            try:
                return self.rm(query_or_queries, k=k, **kwargs)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    print(f"[WARNING] Retrieval timeout/error (attempt {attempt + 1}/{self.max_retries + 1}): {e}. Retrying in {wait}s...", file=sys.stderr)
                    time.sleep(wait)
                else:
                    print(f"[ERROR] Retrieval failed after {self.max_retries + 1} attempts: {e}", file=sys.stderr)
                    raise

    def reset_count(self):
        self.call_count = 0
