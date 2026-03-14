from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


DEFAULT_MODEL = "sroecker/prometheus2"
DEFAULT_URL = "http://127.0.0.1:11434"


def env_flag(primary: str, legacy: str | None = None, default: bool = False) -> bool:
    raw = os.getenv(primary)
    if raw is None and legacy:
        raw = os.getenv(legacy)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def model_name() -> str:
    return os.getenv(
        "INTENT_AUTORESEARCH_LLM_MODEL",
        os.getenv("INTENT_AUTORESEARCH_PROMETHEUS_MODEL", DEFAULT_MODEL),
    ).strip() or DEFAULT_MODEL


def provider_label() -> str:
    model = model_name().lower()
    if "prometheus" in model:
        return "prometheus"
    return "ollama"


def ollama_url() -> str:
    return os.getenv("INTENT_AUTORESEARCH_OLLAMA_URL", DEFAULT_URL).rstrip("/")


def chat(prompt: str, *, system: str | None = None, expect_json: bool = False) -> str:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, object] = {
        "model": model_name(),
        "stream": False,
        "messages": messages,
        "options": {
            "temperature": float(os.getenv("INTENT_AUTORESEARCH_LLM_TEMPERATURE", "0.1")),
            "num_predict": int(os.getenv("INTENT_AUTORESEARCH_LLM_NUM_PREDICT", "192")),
        },
    }
    if expect_json:
        payload["format"] = "json"

    req = urllib.request.Request(
        f"{ollama_url()}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    timeout_sec = float(os.getenv("INTENT_AUTORESEARCH_LLM_TIMEOUT_SEC", "240"))
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"ollama http_error {exc.code}: {body}") from exc
    except Exception as exc:
        raise RuntimeError(f"ollama request failed: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ollama returned invalid JSON: {raw[:400]}") from exc

    message = data.get("message") or {}
    content = str(message.get("content") or "").strip()
    if not content:
        raise RuntimeError("ollama returned empty content")
    return content
