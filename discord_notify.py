from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OPENCLAW_ENV_FILES = (
    Path("/home/sam/projects/Joyortoy/n8n/.env"),
    Path("/home/sam/projects/Joyortoy/n8n/docker/runtime.env"),
)
OPENCLAW_RUNTIME_CONFIG = Path("/home/sam/.openclaw/openclaw.json")
OPENCLAW_RUNTIME_SESSIONS = Path("/home/sam/.openclaw/agents/main/sessions/sessions.json")


def env_file_value(key: str, path: Path) -> str:
    if not path.exists():
        return ""
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        current_key, value = line.split("=", 1)
        if current_key.strip() != key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        return value
    return ""


def discover_discord_webhook() -> str:
    import os

    for key in (
        "INTENT_AUTORESEARCH_DISCORD_WEBHOOK_URL",
        "OPENCLAW_DISCORD_WEBHOOK_URL",
        "DISCORD_WEBHOOK_URL",
    ):
        value = os.getenv(key, "").strip()
        if value:
            return value
    for env_file in OPENCLAW_ENV_FILES:
        for key in (
            "INTENT_AUTORESEARCH_DISCORD_WEBHOOK_URL",
            "OPENCLAW_DISCORD_WEBHOOK_URL",
            "DISCORD_WEBHOOK_URL",
        ):
            value = env_file_value(key, env_file)
            if value:
                return value
    return ""


def discover_discord_bot() -> tuple[str, str]:
    token = ""
    channel_id = ""

    if OPENCLAW_RUNTIME_CONFIG.exists():
        try:
            config = json.loads(OPENCLAW_RUNTIME_CONFIG.read_text(encoding="utf-8"))
            token = str(config.get("channels", {}).get("discord", {}).get("token", "")).strip()
        except Exception:
            token = ""

    if OPENCLAW_RUNTIME_SESSIONS.exists():
        try:
            sessions = json.loads(OPENCLAW_RUNTIME_SESSIONS.read_text(encoding="utf-8"))
            candidates: list[tuple[float, str]] = []
            for value in sessions.values():
                if not isinstance(value, dict) or value.get("channel") != "discord":
                    continue
                updated = float(value.get("updatedAt") or 0)
                group_id = str(value.get("groupId") or "").strip()
                last_to = str(value.get("lastTo") or "").strip()
                target = ""
                if last_to.startswith("channel:"):
                    target = last_to.split(":", 1)[1].strip()
                elif group_id:
                    target = group_id
                if target:
                    candidates.append((updated, target))
            if candidates:
                channel_id = sorted(candidates, key=lambda item: item[0], reverse=True)[0][1]
        except Exception:
            channel_id = ""

    return token, channel_id


def post_json(url: str, payload: dict[str, str], headers: dict[str, str]) -> None:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        resp.read()


def send_message(message: str) -> None:
    webhook = discover_discord_webhook()
    if webhook:
        try:
            post_json(
                webhook,
                {"content": message},
                {
                    "Content-Type": "application/json",
                    "Accept": "*/*",
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
                },
            )
            return
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"discord webhook http_error {exc.code}: {body}") from exc
        except Exception as exc:
            raise RuntimeError(f"discord webhook error: {exc}") from exc

    token, channel_id = discover_discord_bot()
    if not token or not channel_id:
        raise RuntimeError("discord notify skipped: no webhook and no bot/channel config")

    try:
        post_json(
            f"https://discord.com/api/v10/channels/{channel_id}/messages",
            {"content": message},
            {
                "Content-Type": "application/json",
                "Accept": "*/*",
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
                "Authorization": f"Bot {token}",
            },
        )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"discord bot http_error {exc.code}: {body}") from exc
    except Exception as exc:
        raise RuntimeError(f"discord bot error: {exc}") from exc


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: discord_notify.py <message>")
    send_message(sys.argv[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
