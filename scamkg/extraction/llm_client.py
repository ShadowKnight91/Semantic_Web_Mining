"""LLM clients: local Ollama and any OpenAI-compatible endpoint (ASU Voyager), with a disk cache.

Every response is cached under data/cache/llm keyed by (provider, model, temperature, seed,
json mode, messages), so re-running an extraction is free and exactly reproducible.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

from scamkg.utils import get_logger, sha256_text, timestamp

log = get_logger(__name__)


class LLMError(RuntimeError):
    pass


def _post_with_retry(url: str, payload: dict, headers: dict | None = None, timeout: int = 180,
                     retries: int = 4) -> requests.Response:
    """POST with back-off on rate limits / transient server errors. A read timeout is not retried:
    generation is deterministic (fixed seed), so the same request would time out again."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if r.status_code in (429, 502, 503, 504):
                last = LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
                time.sleep(2 ** attempt)
                continue
            return r
        except requests.Timeout as e:
            raise LLMError(f"Request to {url} timed out after {timeout}s") from e
        except requests.ConnectionError as e:
            last = e
            time.sleep(2 ** attempt)
    raise LLMError(f"Request to {url} failed after {retries} attempts: {last}")


class BaseLLMClient:
    provider = "base"

    def __init__(self, model: str, temperature: float = 0.1, seed: int = 42, timeout: int = 180,
                 json_mode: bool = True, max_tokens: int = 2048):
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.timeout = timeout
        self.json_mode = json_mode
        self.max_tokens = max_tokens   # hard cap: small models can loop forever in JSON mode

    def chat(self, messages: list[dict]) -> str:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"provider": self.provider, "model": self.model, "temperature": self.temperature,
                "seed": self.seed, "json_mode": self.json_mode, "max_tokens": self.max_tokens}


class OllamaClient(BaseLLMClient):
    provider = "ollama"

    def __init__(self, model: str, base_url: str = "http://localhost:11434", num_ctx: int = 8192, **kw):
        super().__init__(model, **kw)
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx

    def chat(self, messages: list[dict]) -> str:
        payload = {"model": self.model, "messages": messages, "stream": False,
                   "options": {"temperature": self.temperature, "seed": self.seed, "num_ctx": self.num_ctx,
                               "num_predict": self.max_tokens}}
        if self.json_mode:
            payload["format"] = "json"
        r = _post_with_retry(f"{self.base_url}/api/chat", payload, timeout=self.timeout)
        if r.status_code == 404:
            raise LLMError(f"Ollama model {self.model!r} not found - run `ollama pull {self.model}`")
        if r.status_code >= 400:
            raise LLMError(f"Ollama error {r.status_code}: {r.text[:300]}")
        return r.json()["message"]["content"]


class OpenAICompatibleClient(BaseLLMClient):
    """Chat Completions API client (ASU Voyager, vLLM, LM Studio, OpenAI...)."""

    provider = "openai"

    def __init__(self, model: str, base_url: str, api_key: str | None = None, provider_name: str = "voyager", **kw):
        super().__init__(model, **kw)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.provider = provider_name

    def chat(self, messages: list[dict]) -> str:
        payload = {"model": self.model, "messages": messages, "temperature": self.temperature, "seed": self.seed,
                   "max_tokens": self.max_tokens}
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = self.base_url + "/chat/completions"
        r = _post_with_retry(url, payload, headers=headers, timeout=self.timeout)
        if r.status_code == 400 and self.json_mode and "response_format" in r.text:
            log.warning("Endpoint rejected response_format; retrying without JSON mode")
            self.json_mode = False
            payload.pop("response_format", None)
            r = _post_with_retry(url, payload, headers=headers, timeout=self.timeout)
        if r.status_code >= 400:
            raise LLMError(f"{self.provider} error {r.status_code}: {r.text[:300]}")
        return r.json()["choices"][0]["message"]["content"]


class CachedLLMClient:
    """Wraps a client with a content-addressed disk cache."""

    def __init__(self, client: BaseLLMClient, cache_dir: Path):
        self.client = client
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.last_cached = False

    def __getattr__(self, item):
        return getattr(self.client, item)

    def _key(self, messages: list[dict]) -> str:
        c = self.client
        return sha256_text(json.dumps({"p": c.provider, "m": c.model, "t": c.temperature, "s": c.seed,
                                       "j": c.json_mode, "msg": messages}, sort_keys=True))

    def chat(self, messages: list[dict]) -> str:
        key = self._key(messages)
        path = self.cache_dir / key[:2] / f"{key}.json"
        if path.exists():
            self.last_cached = True
            return json.loads(path.read_text(encoding="utf-8"))["response"]
        self.last_cached = False
        response = self.client.chat(messages)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"response": response, "created": timestamp(), **self.client.describe()}),
                        encoding="utf-8")
        return response

    def describe(self) -> dict:
        return self.client.describe()


def make_client(cfg: dict, cache_dir: Path | None = None, model: str | None = None,
                provider: str | None = None) -> CachedLLMClient | BaseLLMClient:
    llm = cfg["llm"]
    provider = provider or llm["provider"]
    model = model or llm["model"]
    common = dict(temperature=float(llm.get("temperature", 0.1)), seed=int(llm.get("seed", 42)),
                  timeout=int(llm.get("timeout", 180)), json_mode=bool(llm.get("json_mode", True)),
                  max_tokens=int(llm.get("max_tokens", 2048)))
    pcfg = (llm.get("providers") or {}).get(provider, {})
    if provider == "ollama":
        base = os.environ.get(pcfg.get("base_url_env") or "OLLAMA_BASE_URL") or pcfg.get("base_url") \
            or "http://localhost:11434"
        client: BaseLLMClient = OllamaClient(model, base_url=base, num_ctx=int(pcfg.get("num_ctx", 8192)), **common)
    else:
        base = os.environ.get(pcfg.get("base_url_env") or "", "") or pcfg.get("base_url")
        if not base:
            raise LLMError(f"No base URL for provider {provider!r}: set {pcfg.get('base_url_env', 'VOYAGER_BASE_URL')} "
                           "in .env or llm.providers.<name>.base_url in the config")
        key = os.environ.get(pcfg.get("api_key_env") or "", "") or None
        client = OpenAICompatibleClient(model, base_url=base, api_key=key, provider_name=provider, **common)
    if cache_dir is None:
        return client
    return CachedLLMClient(client, Path(cache_dir) / "llm")
