"""Teacher model client (DeepSeek-V3 on SiliconFlow) for data construction, with a response cache and cost accounting.

Every call is cached on disk by a hash of (model, messages, parameters), so a rerun of a data script never pays twice
and its outputs are reproducible from the cache. The API key comes from SILICONFLOW_API_KEY. Calls go direct: proxy
variables are ignored (the endpoint is domestic, and proxy traffic is metered).
"""
import hashlib
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ENDPOINT = "https://api.siliconflow.cn/v1/chat/completions"
MODEL = "deepseek-ai/DeepSeek-V3"
PRICE_PER_MILLION = {"prompt": 2.0, "completion": 8.0}  # CNY, SiliconFlow list price for DeepSeek-V3


class Teacher:
    def __init__(self, cache, model=MODEL, api_key=None):
        self.model, self.cache_path = model, Path(cache)
        self.api_key = api_key or os.environ.get("SILICONFLOW_API_KEY")
        if not self.api_key:
            raise ValueError("SILICONFLOW_API_KEY is required")
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.open(encoding="utf-8"):
                row = json.loads(line)
                self.cache[row["key"]] = row
        self.lock = threading.Lock()
        self.usage = {"requests": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0, "rate_limited": 0}
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # no proxy

    def cost(self):
        return round((self.usage["prompt_tokens"] * PRICE_PER_MILLION["prompt"]
                      + self.usage["completion_tokens"] * PRICE_PER_MILLION["completion"]) / 1e6, 3)

    def chat(self, messages, temperature=0.0, max_tokens=512, json_mode=False, attempts=8):
        params = {"temperature": temperature, "max_tokens": max_tokens, "json_mode": json_mode}
        key = hashlib.sha256(json.dumps([self.model, messages, params], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        with self.lock:
            if key in self.cache:
                self.usage["cached"] += 1
                return self.cache[key]["content"]
        body = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(),
                                         headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        error = None
        for attempt in range(attempts):
            wait = 2 ** min(attempt, 3)
            try:
                with self.opener.open(request, timeout=120) as result:
                    response = json.loads(result.read(), strict=False)
                content = response["choices"][0]["message"]["content"]
                usage = response.get("usage") or {}
                with self.lock:
                    self.usage["requests"] += 1
                    self.usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    self.usage["completion_tokens"] += usage.get("completion_tokens", 0)
                    row = {"key": key, "model": self.model, "params": params, "messages": messages, "content": content,
                           "usage": usage}
                    self.cache[key] = row
                    with self.cache_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                return content
            except urllib.error.HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504}:
                    raise
                error = exc
                if exc.code == 429:
                    with self.lock:
                        self.usage["rate_limited"] += 1
                    retry_after = (exc.headers or {}).get("Retry-After", "")
                    wait = float(retry_after) if retry_after.isdigit() else min(60, 10 * 2 ** attempt)
            except (urllib.error.URLError, TimeoutError, ConnectionError, KeyError, json.JSONDecodeError) as exc:
                error = exc
            time.sleep(wait * random.uniform(1, 1.3))
        raise RuntimeError(f"teacher call failed after {attempts} attempts: {error}")
