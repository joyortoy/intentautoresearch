from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STATUS_DIR = ROOT / "logs" / "status"
LEGACY_QWEN_ROOT = Path(os.getenv("AUTORESEARCH_QWEN_ROOT", "/home/sam/projects/autoresearch"))
INTENTSTACK_URL = os.getenv("INTENTSTACK_URL", "http://127.0.0.1:8090").rstrip("/")
SERVICE_NAMES = (
    "intentautoresearch-loop.service",
    "autoresearch-loop.service",
)


def status_path(name: str) -> Path:
    return STATUS_DIR / f"{name}.json"


def load_status(name: str) -> dict:
    path = status_path(name)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_status(args: argparse.Namespace) -> int:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    payload = load_status(args.name)
    payload.update(
        {
            "name": args.name,
            "state": args.state,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
    )
    for key in (
        "iteration",
        "pid",
        "iteration_log",
        "results_tsv",
        "deck_path",
        "status_line",
        "summary",
    ):
        value = getattr(args, key)
        if value is not None:
            payload[key] = value
    path = status_path(args.name)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)
    return 0


def render_status(name: str) -> str:
    payload = load_status(name)
    if not payload:
        return f"{name}: no status file"
    lines = [
        f"{name}: state={payload.get('state', '-')}",
        f"updated_at={payload.get('updated_at', '-')}",
    ]
    if payload.get("iteration") is not None:
        lines.append(f"iteration={payload.get('iteration')}")
    if payload.get("status_line"):
        lines.append(f"best={payload['status_line']}")
    if payload.get("summary"):
        lines.append("summary:")
        lines.extend(str(payload["summary"]).splitlines())
    if payload.get("iteration_log"):
        lines.append(f"log={payload['iteration_log']}")
    return "\n".join(lines)


def show_status(args: argparse.Namespace) -> int:
    names = [args.name] if args.name else ["complex", "good"]
    print("\n\n".join(render_status(name) for name in names))
    return 0


def run_command(command: list[str]) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        return False, str(exc)
    output = (completed.stdout or completed.stderr or "").strip()
    return completed.returncode == 0, output


def fetch_json(url: str) -> tuple[bool, dict | str]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=4) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
        return False, str(exc)
    return True, payload


def read_tsv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))
    except Exception:
        return []


def status_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = (row.get("status") or "").strip() or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return counts


def summarize_intent_results(path: Path) -> dict:
    rows = read_tsv_rows(path)
    kept = [row for row in rows if (row.get("status") or "").strip() == "keep"]
    best_keep = None
    if kept:
        best_keep = max(
            kept,
            key=lambda row: (
                float(row.get("val_macro_f1") or "0"),
                float(row.get("val_accuracy") or "0"),
                -float(row.get("val_intent_loss") or "inf"),
            ),
        )
    return {
        "rows": len(rows),
        "status_counts": status_counts(rows),
        "best_keep": best_keep,
        "latest": rows[-1] if rows else None,
    }


def summarize_qwen_results(path: Path) -> dict:
    rows = read_tsv_rows(path)
    kept: list[dict] = []
    for row in rows:
        if (row.get("status") or "").strip() != "keep":
            continue
        try:
            row["_val_bpb"] = float(row.get("val_bpb") or "inf")
        except Exception:
            continue
        kept.append(row)
    best_keep = min(kept, key=lambda row: row["_val_bpb"]) if kept else None
    return {
        "rows": len(rows),
        "status_counts": status_counts(rows),
        "best_keep": best_keep,
        "latest": rows[-1] if rows else None,
    }


def latest_qwen_eval() -> dict:
    reports = sorted((LEGACY_QWEN_ROOT / "logs").glob("qwen*_retrieval_eval.json"))
    if not reports:
        return {}
    path = reports[-1]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"path": str(path), "error": "unreadable"}

    best = None
    for run in payload.get("runs", []):
        summary = run.get("modes", {}).get("query", {}).get("summary", {})
        row = (
            float(summary.get("mrr") or 0.0),
            run.get("config_label", ""),
            summary,
        )
        if best is None or row[0] > best[0]:
            best = row
    if best is None:
        return {"path": str(path)}
    return {
        "path": str(path),
        "config_label": best[1],
        "summary": best[2],
    }


def format_intent_row(row: dict | None) -> str:
    if not row:
        return "none"
    return (
        f"{row.get('description', '-')}"
        f" | loss={float(row.get('val_intent_loss') or 'inf'):.6f}"
        f" | acc={float(row.get('val_accuracy') or '0'):.4f}"
        f" | f1={float(row.get('val_macro_f1') or '0'):.4f}"
        f" | mem={float(row.get('memory_gb') or '0'):.3f} GB"
        f" | status={row.get('status', '-')}"
    )


def format_qwen_row(row: dict | None) -> str:
    if not row:
        return "none"
    return (
        f"{row.get('description', '-')}"
        f" | val_bpb={float(row.get('val_bpb') or 'inf'):.6f}"
        f" | mem={float(row.get('memory_gb') or '0'):.1f} GB"
        f" | status={row.get('status', '-')}"
    )


def render_service_status() -> list[str]:
    lines = ["Service"]
    unavailable = False
    for name in SERVICE_NAMES:
        active_ok, active = run_command(["systemctl", "--user", "is-active", name])
        enabled_ok, enabled = run_command(["systemctl", "--user", "is-enabled", name])
        raw = " ".join(part for part in (active, enabled) if part)
        if "Failed to connect to user scope bus" in raw:
            unavailable = True
            continue
        if not (active_ok or active or enabled_ok or enabled):
            continue
        lines.append(
            f"{name}: active={active or 'unknown'} enabled={enabled or 'unknown'}"
        )
    if unavailable:
        lines.append("service state unavailable from sandbox; run on host for live systemd state")
    elif len(lines) == 1:
        lines.append("service state unavailable")
    return lines


def render_intentstack_health() -> list[str]:
    ok, payload = fetch_json(f"{INTENTSTACK_URL}/admin/health")
    lines = ["IntentStack"]
    if not ok:
        lines.append(f"health: unavailable ({payload})")
        return lines
    models = payload.get("models") or {}
    lines.append(
        "health: status={status} sqlite={sqlite_ok} ollama={ollama_ok}".format(
            status=payload.get("status", "?"),
            sqlite_ok=(payload.get("sqlite") or {}).get("ok"),
            ollama_ok=(payload.get("ollama") or {}).get("ok"),
        )
    )
    lines.append(
        "models: intent={intent} memory_agent={memory} chat={chat} judge={judge}".format(
            intent=models.get("intent", "-"),
            memory=models.get("memory_agent", "-"),
            chat=models.get("chat", "-"),
            judge=models.get("judge", "-"),
        )
    )
    return lines


def render_intent_repo_section() -> list[str]:
    lines = ["Intent Repo"]
    for name, path in (
        ("complex", ROOT / "complex_results.tsv"),
        ("good", ROOT / "good_results.tsv"),
    ):
        summary = summarize_intent_results(path)
        lines.append(
            f"{name}: rows={summary['rows']} status_counts={summary['status_counts'] or '{}'}"
        )
        lines.append(f"{name} best_keep: {format_intent_row(summary['best_keep'])}")
        lines.append(f"{name} latest: {format_intent_row(summary['latest'])}")
        lines.append(render_status(name))
    return lines


def render_qwen_repo_section() -> list[str]:
    lines = ["Legacy Qwen Repo", f"root={LEGACY_QWEN_ROOT}"]
    summary = summarize_qwen_results(LEGACY_QWEN_ROOT / "results.tsv")
    lines.append(f"rows={summary['rows']} status_counts={summary['status_counts'] or '{}'}")
    lines.append(f"best_keep: {format_qwen_row(summary['best_keep'])}")
    lines.append(f"latest: {format_qwen_row(summary['latest'])}")

    qwen_eval = latest_qwen_eval()
    if qwen_eval:
        summary_bits = qwen_eval.get("summary") or {}
        lines.append(
            "latest_qwen_retrieval: config={cfg} hit@1={hit1} hit@5={hitk} mrr={mrr} path={path}".format(
                cfg=qwen_eval.get("config_label", "-"),
                hit1=summary_bits.get("hit_at_1", "-"),
                hitk=summary_bits.get("hit_at_k", "-"),
                mrr=summary_bits.get("mrr", "-"),
                path=qwen_eval.get("path", "-"),
            )
        )
    else:
        lines.append("latest_qwen_retrieval: none")
    return lines


def report_status(_: argparse.Namespace) -> int:
    sections = [
        render_service_status(),
        render_intentstack_health(),
        render_intent_repo_section(),
        render_qwen_repo_section(),
    ]
    print("\n\n".join("\n".join(section) for section in sections))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    write_parser = subparsers.add_parser("write")
    write_parser.add_argument("--name", required=True)
    write_parser.add_argument("--state", required=True)
    write_parser.add_argument("--iteration", type=int)
    write_parser.add_argument("--pid", type=int)
    write_parser.add_argument("--iteration-log")
    write_parser.add_argument("--results-tsv")
    write_parser.add_argument("--deck-path")
    write_parser.add_argument("--status-line")
    write_parser.add_argument("--summary")
    write_parser.set_defaults(func=write_status)

    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("--name")
    show_parser.set_defaults(func=show_status)

    report_parser = subparsers.add_parser("report")
    report_parser.set_defaults(func=report_status)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
