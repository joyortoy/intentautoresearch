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


def default_num_predict(expect_json: bool) -> int:
    if expect_json:
        return int(
            os.getenv(
                "INTENT_AUTORESEARCH_LLM_JSON_NUM_PREDICT",
                os.getenv("INTENT_AUTORESEARCH_LLM_NUM_PREDICT", "512"),
            )
        )
    return int(os.getenv("INTENT_AUTORESEARCH_LLM_NUM_PREDICT", "192"))


def json_max_num_predict() -> int:
    return int(os.getenv("INTENT_AUTORESEARCH_LLM_JSON_MAX_NUM_PREDICT", "1024"))


def chat(
    prompt: str,
    *,
    system: str | None = None,
    expect_json: bool = False,
    num_predict: int | None = None,
) -> str:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    timeout_sec = float(os.getenv("INTENT_AUTORESEARCH_LLM_TIMEOUT_SEC", "240"))
    current_num_predict = num_predict or default_num_predict(expect_json)
    max_attempts = 2 if expect_json else 1
    last_content = ""
    last_done_reason = ""

    for attempt in range(max_attempts):
        payload: dict[str, object] = {
            "model": model_name(),
            "stream": False,
            "messages": messages,
            "options": {
                "temperature": float(os.getenv("INTENT_AUTORESEARCH_LLM_TEMPERATURE", "0.1")),
                "num_predict": current_num_predict,
            },
        }
        if expect_json:
            payload["format"] = "json"

        req = urllib.request.Request(
            f"{ollama_url()}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
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
        done_reason = str(data.get("done_reason") or "")
        if not content:
            raise RuntimeError("ollama returned empty content")
        if not expect_json or done_reason != "length":
            return content
        last_content = content
        last_done_reason = done_reason
        if attempt + 1 >= max_attempts:
            break
        next_num_predict = min(current_num_predict * 2, json_max_num_predict())
        if next_num_predict <= current_num_predict:
            break
        current_num_predict = next_num_predict

    preview = last_content[:400]
    raise RuntimeError(
        f"ollama JSON response truncated (done_reason={last_done_reason or 'unknown'}, "
        f"num_predict={current_num_predict}): {preview!r}"
    )
