#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_SCRIPT="${ROOT_DIR}/run_complex_intent_sweep.sh"
DECK_BUILDER="${ROOT_DIR}/build_complex_deck.py"
DECK_AUGMENTER="${ROOT_DIR}/augment_complex_deck.py"
SUMMARY_SCRIPT="${ROOT_DIR}/prometheus_progress_summary.py"
LOG_DIR="${ROOT_DIR}/logs"
LOOP_LOG="${LOG_DIR}/complex_intent_forever.log"
ITERATION_LOG_DIR="${LOG_DIR}/complex_forever_iterations"
SLEEP_SEC="${INTENT_AUTORESEARCH_FOREVER_SLEEP_SEC:-5}"
RESULTS_TSV="${ROOT_DIR}/complex_results.tsv"
DECK_PATH="${ROOT_DIR}/data/complex_intent_deck.jsonl"
TARGET_ACCURACY="${INTENT_AUTORESEARCH_TARGET_ACCURACY:-1.0}"
TARGET_MACRO_F1="${INTENT_AUTORESEARCH_TARGET_MACRO_F1:-1.0}"
NOTIFY_EVERY="${INTENT_AUTORESEARCH_NOTIFY_EVERY:-10}"
AUTO_EXPAND_ON_IMPROVEMENT="${INTENT_AUTORESEARCH_AUTO_EXPAND_ON_IMPROVEMENT:-0}"
AUTO_EXPAND_ON_STALL="${INTENT_AUTORESEARCH_AUTO_EXPAND_ON_STALL:-1}"
AUTO_EXPAND_MAX_STAGE="${INTENT_AUTORESEARCH_AUTO_EXPAND_MAX_STAGE:-4}"
BEST_STATE_JSON="${ROOT_DIR}/.complex_best_state.json"
DECK_STAGE_FILE="${ROOT_DIR}/.complex_deck_stage"
OPENCLAW_ENV_FILES=(
  "/home/sam/projects/Joyortoy/n8n/.env"
  "/home/sam/projects/Joyortoy/n8n/docker/runtime.env"
)
OPENCLAW_RUNTIME_CONFIG="/home/sam/.openclaw/openclaw.json"
OPENCLAW_RUNTIME_SESSIONS="/home/sam/.openclaw/agents/main/sessions/sessions.json"

env_file_value() {
  local key="$1"
  local path="$2"
  python3 - "$key" "$path" <<'PY'
import pathlib
import sys

key = sys.argv[1]
path = pathlib.Path(sys.argv[2])
if not path.exists():
    raise SystemExit(1)
for raw in path.read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    current_key, value = line.split("=", 1)
    if current_key.strip() != key:
        continue
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    print(value)
    raise SystemExit(0)
raise SystemExit(1)
PY
}

discover_discord_webhook() {
  local value=""
  if [[ -n "${INTENT_AUTORESEARCH_DISCORD_WEBHOOK_URL:-}" ]]; then
    printf '%s\n' "${INTENT_AUTORESEARCH_DISCORD_WEBHOOK_URL}"
    return 0
  fi
  if [[ -n "${OPENCLAW_DISCORD_WEBHOOK_URL:-}" ]]; then
    printf '%s\n' "${OPENCLAW_DISCORD_WEBHOOK_URL}"
    return 0
  fi
  if [[ -n "${DISCORD_WEBHOOK_URL:-}" ]]; then
    printf '%s\n' "${DISCORD_WEBHOOK_URL}"
    return 0
  fi
  for env_file in "${OPENCLAW_ENV_FILES[@]}"; do
    value="$(env_file_value "INTENT_AUTORESEARCH_DISCORD_WEBHOOK_URL" "${env_file}" 2>/dev/null || true)"
    if [[ -n "${value}" ]]; then
      printf '%s\n' "${value}"
      return 0
    fi
    value="$(env_file_value "OPENCLAW_DISCORD_WEBHOOK_URL" "${env_file}" 2>/dev/null || true)"
    if [[ -n "${value}" ]]; then
      printf '%s\n' "${value}"
      return 0
    fi
    value="$(env_file_value "DISCORD_WEBHOOK_URL" "${env_file}" 2>/dev/null || true)"
    if [[ -n "${value}" ]]; then
      printf '%s\n' "${value}"
      return 0
    fi
  done
  return 1
}

DISCORD_WEBHOOK_URL="$(discover_discord_webhook || true)"

discover_discord_bot() {
  python3 - "${OPENCLAW_RUNTIME_CONFIG}" "${OPENCLAW_RUNTIME_SESSIONS}" <<'PY'
import json
import pathlib
import sys

config_path = pathlib.Path(sys.argv[1])
sessions_path = pathlib.Path(sys.argv[2])

token = ""
channel_id = ""

if config_path.exists():
    try:
        config = json.loads(config_path.read_text())
        token = (
            config.get("channels", {})
            .get("discord", {})
            .get("token", "")
            .strip()
        )
    except Exception:
        token = ""

if sessions_path.exists():
    try:
        sessions = json.loads(sessions_path.read_text())
        candidates = []
        for value in sessions.values():
            if not isinstance(value, dict):
                continue
            if value.get("channel") != "discord":
                continue
            updated = value.get("updatedAt") or 0
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

if token and channel_id:
    print(token)
    print(channel_id)
PY
}

DISCORD_BOT_TOKEN=""
DISCORD_CHANNEL_ID=""
if [[ -z "${DISCORD_WEBHOOK_URL}" ]]; then
  mapfile -t _discord_bot_lines < <(discover_discord_bot || true)
  if (( ${#_discord_bot_lines[@]} >= 2 )); then
    DISCORD_BOT_TOKEN="${_discord_bot_lines[0]}"
    DISCORD_CHANNEL_ID="${_discord_bot_lines[1]}"
  fi
fi

mkdir -p "${LOG_DIR}"
mkdir -p "${ITERATION_LOG_DIR}"

send_discord() {
  local message="$1"
  if [[ -n "${DISCORD_WEBHOOK_URL}" ]]; then
    python3 - "${DISCORD_WEBHOOK_URL}" "${message}" <<'PY'
import json
import sys
import urllib.request
import urllib.error

url = sys.argv[1]
message = sys.argv[2]
payload = json.dumps({"content": message}).encode("utf-8")
req = urllib.request.Request(
    url,
    data=payload,
    headers={
        "Content-Type": "application/json",
        "Accept": "*/*",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
    },
)
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        resp.read()
except urllib.error.HTTPError as exc:
    body = exc.read().decode("utf-8", "replace")
    print(f"discord webhook http_error {exc.code}: {body}", file=sys.stderr)
    raise
except Exception as exc:
    print(f"discord webhook error: {exc}", file=sys.stderr)
    raise
PY
    return 0
  fi
  if [[ -z "${DISCORD_BOT_TOKEN}" || -z "${DISCORD_CHANNEL_ID}" ]]; then
    echo "discord notify skipped: no webhook and no bot/channel config" >&2
    return 0
  fi
  python3 - "${DISCORD_BOT_TOKEN}" "${DISCORD_CHANNEL_ID}" "${message}" <<'PY'
import json
import sys
import urllib.request
import urllib.error

token = sys.argv[1]
channel_id = sys.argv[2]
message = sys.argv[3]
payload = json.dumps({"content": message}).encode("utf-8")
req = urllib.request.Request(
    f"https://discord.com/api/v10/channels/{channel_id}/messages",
    data=payload,
    headers={
        "Content-Type": "application/json",
        "Accept": "*/*",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
        "Authorization": f"Bot {token}",
    },
)
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        resp.read()
except urllib.error.HTTPError as exc:
    body = exc.read().decode("utf-8", "replace")
    print(f"discord bot http_error {exc.code}: {body}", file=sys.stderr)
    raise
except Exception as exc:
    print(f"discord bot error: {exc}", file=sys.stderr)
    raise
PY
}

best_summary() {
  python3 - "${RESULTS_TSV}" <<'PY'
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1], newline=""), delimiter="\t"))
kept = [row for row in rows if row.get("status") == "keep"]
if not kept:
    print("no-keep")
    raise SystemExit(0)
best = max(
    kept,
    key=lambda row: (
        float(row.get("val_macro_f1") or "0"),
        float(row.get("val_accuracy") or "0"),
        -float(row.get("val_intent_loss") or "inf"),
    ),
)
print(
    f"{best.get('description','-')} | "
    f"acc={float(best.get('val_accuracy') or '0'):.4f} | "
    f"f1={float(best.get('val_macro_f1') or '0'):.4f} | "
    f"loss={float(best.get('val_intent_loss') or 'inf'):.6f}"
)
PY
}

llm_notify_summary() {
  local iteration="$1"
  local tmp_out
  local tmp_err
  tmp_out="$(mktemp)"
  tmp_err="$(mktemp)"
  if python3 "${SUMMARY_SCRIPT}" "${iteration}" "${RESULTS_TSV}" "${ITERATION_LOG_DIR}" >"${tmp_out}" 2>"${tmp_err}"; then
    cat "${tmp_out}"
  else
    cat "${tmp_out}"
  fi
  if [[ -s "${tmp_err}" ]]; then
    printf '[%s] summary_artifact=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$(tail -n 1 "${tmp_err}")" | tee -a "${LOOP_LOG}" >/dev/null
  fi
  rm -f "${tmp_out}" "${tmp_err}"
}

current_best_json() {
  python3 - "${RESULTS_TSV}" <<'PY'
import csv
import json
import sys

rows = list(csv.DictReader(open(sys.argv[1], newline=""), delimiter="\t"))
kept = [row for row in rows if row.get("status") == "keep"]
if not kept:
    print("{}")
    raise SystemExit(0)
best = max(
    kept,
    key=lambda row: (
        float(row.get("val_macro_f1") or "0"),
        float(row.get("val_accuracy") or "0"),
        -float(row.get("val_intent_loss") or "inf"),
    ),
)
payload = {
    "description": best.get("description", "-"),
    "val_accuracy": float(best.get("val_accuracy") or "0"),
    "val_macro_f1": float(best.get("val_macro_f1") or "0"),
    "val_intent_loss": float(best.get("val_intent_loss") or "inf"),
}
print(json.dumps(payload, separators=(",", ":")))
PY
}

deck_expansion_signal() {
  local current_json="$1"
  python3 - "${BEST_STATE_JSON}" "${current_json}" <<'PY'
import json
import pathlib
import sys

state_path = pathlib.Path(sys.argv[1])
current = json.loads(sys.argv[2])
previous = {}
if state_path.exists():
    try:
        previous = json.loads(state_path.read_text())
    except Exception:
        previous = {}

def score(row):
    return (
        float(row.get("val_macro_f1") or 0.0),
        float(row.get("val_accuracy") or 0.0),
        -float(row.get("val_intent_loss") or float("inf")),
    )

if not current:
    print("")
    raise SystemExit(0)

improved = score(current) > score(previous)
state_path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
print("improvement" if improved else "stall")
PY
}

deck_rows() {
  python3 - "${DECK_PATH}" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.exists():
    print("0")
    raise SystemExit(0)
print(sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()))
PY
}

expand_deck_once() {
  local current_stage=0
  if [[ -f "${DECK_STAGE_FILE}" ]]; then
    current_stage="$(cat "${DECK_STAGE_FILE}")"
  fi
  if (( current_stage >= AUTO_EXPAND_MAX_STAGE )); then
    printf '[%s] deck expansion skipped: max stage reached (%s)\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${current_stage}" | tee -a "${LOOP_LOG}"
    return 0
  fi
  local next_stage=$((current_stage + 1))
  printf '[%s] deck expansion triggered: stage=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${next_stage}" | tee -a "${LOOP_LOG}"
  python3 "${DECK_BUILDER}" --all-reports >>"${LOOP_LOG}" 2>&1 || true
  python3 "${DECK_AUGMENTER}" --stage "${next_stage}" >>"${LOOP_LOG}" 2>&1
  printf '%s\n' "${next_stage}" > "${DECK_STAGE_FILE}"
  printf '[%s] deck rows=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$(deck_rows)" | tee -a "${LOOP_LOG}"
}

iteration=0
while true; do
  iteration=$((iteration + 1))
  iteration_log="${ITERATION_LOG_DIR}/complex_forever_iteration_${iteration}_$(date '+%Y%m%d_%H%M%S').log"
  printf '\n[%s] complex-forever iteration=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${iteration}" | tee -a "${LOOP_LOG}"
  printf '[%s] iteration_log=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${iteration_log}" | tee -a "${LOOP_LOG}"
  bash "${SWEEP_SCRIPT}" >"${iteration_log}" 2>&1
  status_line="$(python3 - "${RESULTS_TSV}" "${TARGET_ACCURACY}" "${TARGET_MACRO_F1}" <<'PY'
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1], newline=""), delimiter="\t"))
target_acc = float(sys.argv[2])
target_f1 = float(sys.argv[3])
kept = [row for row in rows if row.get("status") == "keep"]
if not kept:
    print("no-keep")
    raise SystemExit(0)
best = max(
    kept,
    key=lambda row: (
        float(row.get("val_macro_f1") or "0"),
        float(row.get("val_accuracy") or "0"),
        -float(row.get("val_intent_loss") or "inf"),
    ),
)
acc = float(best.get("val_accuracy") or "0")
f1 = float(best.get("val_macro_f1") or "0")
print(f"{best.get('description','-')}\t{acc:.4f}\t{f1:.4f}")
if acc >= target_acc and f1 >= target_f1:
    raise SystemExit(10)
PY
  )" || rc=$?
  rc="${rc:-0}"
  printf '[%s] best=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${status_line}" | tee -a "${LOOP_LOG}"
  current_json="$(current_best_json)"
  expand_reason="$(deck_expansion_signal "${current_json}")"
  should_expand=0
  if [[ "${expand_reason}" == "improvement" && "${AUTO_EXPAND_ON_IMPROVEMENT}" == "1" ]]; then
    should_expand=1
  fi
  if [[ "${expand_reason}" == "stall" && "${AUTO_EXPAND_ON_STALL}" == "1" ]]; then
    should_expand=1
  fi
  if [[ "${should_expand}" == "1" ]]; then
    printf '[%s] deck expansion reason=%s current=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${expand_reason}" "${current_json}" | tee -a "${LOOP_LOG}"
    expand_deck_once
  fi
  if [[ "${rc}" -eq 10 ]]; then
    printf '[%s] stopping: clean threshold reached\n' "$(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOOP_LOG}"
    break
  fi
  if (( iteration % NOTIFY_EVERY == 0 )); then
    summary="$(llm_notify_summary "${iteration}")"
    if [[ -z "${summary}" ]]; then
      summary="$(best_summary)"
    fi
    note="${summary}"
    printf '[%s] notify=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${note}" | tee -a "${LOOP_LOG}"
    send_discord "${note}" 2>>"${LOOP_LOG}" || printf '[%s] discord notify failed\n' "$(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOOP_LOG}"
  fi
  sleep "${SLEEP_SEC}"
done
