#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "bad_intent_probe"
DEFAULT_OUT_PATH = PROJECT_ROOT / "data" / "complex_intent_deck.jsonl"


def latest_report(report_dir: Path) -> Path:
    candidates = sorted(report_dir.glob("bad_intent_probe_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No probe reports found in {report_dir}")
    return candidates[-1]


def load_rows(report_path: Path) -> list[dict]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    rows = []
    for row in payload.get("results", []):
        if row.get("assessment", {}).get("outcome") != "pass":
            continue
        prompt = str(row.get("prompt", "")).strip()
        expected_intent = str(row.get("routed", {}).get("intent", "")).strip()
        if not prompt or not expected_intent:
            continue
        rows.append(
            {
                "prompt": prompt,
                "expected_intent": expected_intent,
                "note": f"expanded complex deck from {report_path.name} [{row.get('spec', '-')}]",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the complex intent deck from the latest passing probe results.")
    parser.add_argument("--report", default="", help="Explicit bad_intent_probe JSON report path")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR), help="Directory containing bad intent probe reports")
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH), help="Output JSONL path")
    parser.add_argument("--all-reports", action="store_true", help="Merge passing rows from all probe reports in report-dir")
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    if args.report:
        report_paths = [Path(args.report)]
    elif args.all_reports:
        report_paths = sorted(report_dir.glob("bad_intent_probe_*.json"))
        if not report_paths:
            raise SystemExit(f"No probe reports found in {report_dir}")
    else:
        report_paths = [latest_report(report_dir)]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    seen: set[str] = set()
    for report_path in report_paths:
        for row in load_rows(report_path):
            key = f"{row['expected_intent']}\n{row['prompt']}"
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    if not rows:
        raise SystemExit("No passing rows found in selected reports")

    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"reports={len(report_paths)}")
    print(f"out={out_path}")
    print(f"rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
