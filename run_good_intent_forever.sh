#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_SCRIPT="${ROOT_DIR}/run_good_intent_sweep.sh"
DECK_BUILDER="${ROOT_DIR}/build_good_intent_deck.py"
DECK_AUGMENTER="${ROOT_DIR}/augment_good_intent_deck.py"
SUMMARY_SCRIPT="${ROOT_DIR}/prometheus_progress_summary.py"
STATUS_SCRIPT="${ROOT_DIR}/autoresearch_status.py"
LOG_DIR="${ROOT_DIR}/logs"
LOOP_LOG="${LOG_DIR}/good_intent_forever.log"
ITERATION_LOG_DIR="${LOG_DIR}/good_forever_iterations"
SLEEP_SEC="${INTENT_AUTORESEARCH_GOOD_FOREVER_SLEEP_SEC:-5}"
SWEEP_TIMEOUT_SEC="${INTENT_AUTORESEARCH_GOOD_SWEEP_TIMEOUT_SEC:-600}"
RESULTS_TSV="${ROOT_DIR}/good_results.tsv"
BEST_CONFIG_JSON="${ROOT_DIR}/good_best_config.json"
DECK_PATH="${ROOT_DIR}/data/good_intent_clean_deck.jsonl"
TARGET_ACCURACY="${INTENT_AUTORESEARCH_GOOD_TARGET_ACCURACY:-1.0}"
TARGET_MACRO_F1="${INTENT_AUTORESEARCH_GOOD_TARGET_MACRO_F1:-1.0}"
NOTIFY_EVERY="${INTENT_AUTORESEARCH_GOOD_NOTIFY_EVERY:-10}"
AUTO_EXPAND_ON_IMPROVEMENT="${INTENT_AUTORESEARCH_GOOD_AUTO_EXPAND_ON_IMPROVEMENT:-0}"
AUTO_EXPAND_ON_STALL="${INTENT_AUTORESEARCH_GOOD_AUTO_EXPAND_ON_STALL:-1}"
AUTO_EXPAND_MAX_STAGE="${INTENT_AUTORESEARCH_GOOD_AUTO_EXPAND_MAX_STAGE:-0}"
STALL_DATASET_GROWTH_ROWS="${INTENT_AUTORESEARCH_GOOD_STALL_DATASET_GROWTH_ROWS:-48}"
AUTO_ADJUST_MODEL_ON_STALL="${INTENT_AUTORESEARCH_GOOD_AUTO_ADJUST_MODEL_ON_STALL:-1}"
AUTO_ADJUST_MODEL_MAX_STAGE="${INTENT_AUTORESEARCH_GOOD_AUTO_ADJUST_MODEL_MAX_STAGE:-0}"
EXPAND_EVERY_ITERATION="${INTENT_AUTORESEARCH_GOOD_EXPAND_EVERY_ITERATION:-0}"
RESET_TO_BEST_ON_CRASH="${INTENT_AUTORESEARCH_GOOD_RESET_TO_BEST_ON_CRASH:-1}"
BEST_STATE_JSON="${ROOT_DIR}/.good_best_state.json"
DECK_STAGE_FILE="${ROOT_DIR}/.good_deck_stage"
MODEL_STAGE_FILE="${ROOT_DIR}/.good_model_stage"
RECOVERY_FLAG_FILE="${ROOT_DIR}/.best_only_recovery.json"
SEARCH_ARCHIVE_DIR="${LOG_DIR}/good_curriculum_archives"

mkdir -p "${LOG_DIR}" "${ITERATION_LOG_DIR}"

read_stage_value() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    cat "${path}"
  else
    printf '0\n'
  fi
}

archive_search_state() {
  local reason="$1"
  local stamp safe_reason archive_prefix
  stamp="$(date '+%Y%m%d_%H%M%S')"
  safe_reason="$(printf '%s' "${reason}" | tr -cs 'A-Za-z0-9_-' '_')"
  archive_prefix="${SEARCH_ARCHIVE_DIR}/${stamp}_${safe_reason}"
  mkdir -p "${SEARCH_ARCHIVE_DIR}"
  [[ -f "${RESULTS_TSV}" ]] && mv "${RESULTS_TSV}" "${archive_prefix}_results.tsv"
  [[ -f "${BEST_CONFIG_JSON}" ]] && mv "${BEST_CONFIG_JSON}" "${archive_prefix}_best_config.json"
  [[ -f "${BEST_STATE_JSON}" ]] && mv "${BEST_STATE_JSON}" "${archive_prefix}_best_state.json"
  rm -f "${RECOVERY_FLAG_FILE}"
  printf '[%s] search state reset: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${archive_prefix}" | tee -a "${LOOP_LOG}"
}

can_expand_deck() {
  local current_stage=0
  current_stage="$(read_stage_value "${DECK_STAGE_FILE}")"
  if (( AUTO_EXPAND_MAX_STAGE > 0 && current_stage >= AUTO_EXPAND_MAX_STAGE )); then
    return 1
  fi
  return 0
}

can_adjust_model() {
  local current_stage=0
  current_stage="$(read_stage_value "${MODEL_STAGE_FILE}")"
  if (( AUTO_ADJUST_MODEL_MAX_STAGE > 0 && current_stage >= AUTO_ADJUST_MODEL_MAX_STAGE )); then
    return 1
  fi
  return 0
}

write_status() {
  local state="$1"
  local args=(
    write
    --name "good"
    --state "${state}"
    --pid "$$"
    --results-tsv "${RESULTS_TSV}"
    --deck-path "${DECK_PATH}"
    --status-line "${status_line:-}"
    --summary "${summary:-}"
  )
  if [[ -n "${iteration:-}" ]]; then
    args+=(--iteration "${iteration}")
  fi
  if [[ -n "${iteration_log:-}" ]]; then
    args+=(--iteration-log "${iteration_log}")
  fi
  python3 "${STATUS_SCRIPT}" "${args[@]}" >/dev/null 2>&1 || true
}

status_message() {
  python3 "${STATUS_SCRIPT}" show --name "good"
}

if [[ ! -f "${DECK_PATH}" ]]; then
  python3 "${DECK_BUILDER}" >>"${LOOP_LOG}" 2>&1
fi

current_best_json() {
  python3 - "${RESULTS_TSV}" <<'PY'
import csv
import json
import sys

rows = []
try:
    with open(sys.argv[1], newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
except FileNotFoundError:
    print("")
    raise SystemExit(0)

kept = [row for row in rows if row.get("status") == "keep"]
if not kept:
    print("")
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
    "description": best.get("description", ""),
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

apply_model_stage_overrides() {
  local model_stage=0
  if [[ -f "${MODEL_STAGE_FILE}" ]]; then
    model_stage="$(cat "${MODEL_STAGE_FILE}")"
  fi

  unset INTENT_AUTORESEARCH_STAGNATION_MIN_EMBED_DIM
  unset INTENT_AUTORESEARCH_STAGNATION_MIN_HIDDEN_DIM
  unset INTENT_AUTORESEARCH_STAGNATION_MIN_MAX_LEN
  unset INTENT_AUTORESEARCH_STAGNATION_MIN_TIME_BUDGET

  case "${model_stage}" in
    0)
      ;;
    1)
      export INTENT_AUTORESEARCH_STAGNATION_MIN_EMBED_DIM=160
      export INTENT_AUTORESEARCH_STAGNATION_MIN_HIDDEN_DIM=224
      export INTENT_AUTORESEARCH_STAGNATION_MIN_MAX_LEN=64
      export INTENT_AUTORESEARCH_STAGNATION_MIN_TIME_BUDGET=30
      ;;
    2)
      export INTENT_AUTORESEARCH_STAGNATION_MIN_EMBED_DIM=192
      export INTENT_AUTORESEARCH_STAGNATION_MIN_HIDDEN_DIM=256
      export INTENT_AUTORESEARCH_STAGNATION_MIN_MAX_LEN=64
      export INTENT_AUTORESEARCH_STAGNATION_MIN_TIME_BUDGET=30
      ;;
    3)
      export INTENT_AUTORESEARCH_STAGNATION_MIN_EMBED_DIM=224
      export INTENT_AUTORESEARCH_STAGNATION_MIN_HIDDEN_DIM=320
      export INTENT_AUTORESEARCH_STAGNATION_MIN_MAX_LEN=80
      export INTENT_AUTORESEARCH_STAGNATION_MIN_TIME_BUDGET=45
      ;;
    *)
      export INTENT_AUTORESEARCH_STAGNATION_MIN_EMBED_DIM=224
      export INTENT_AUTORESEARCH_STAGNATION_MIN_HIDDEN_DIM=320
      export INTENT_AUTORESEARCH_STAGNATION_MIN_MAX_LEN=80
      export INTENT_AUTORESEARCH_STAGNATION_MIN_TIME_BUDGET=60
      ;;
  esac

  if (( model_stage > 0 )); then
    printf '[%s] model stage=%s embed>=%s hidden>=%s max_len>=%s budget>=%s\n' \
      "$(date '+%Y-%m-%d %H:%M:%S')" \
      "${model_stage}" \
      "${INTENT_AUTORESEARCH_STAGNATION_MIN_EMBED_DIM}" \
      "${INTENT_AUTORESEARCH_STAGNATION_MIN_HIDDEN_DIM}" \
      "${INTENT_AUTORESEARCH_STAGNATION_MIN_MAX_LEN}" \
      "${INTENT_AUTORESEARCH_STAGNATION_MIN_TIME_BUDGET}" | tee -a "${LOOP_LOG}"
  fi
}

expand_deck_once() {
  local current_stage=0
  local current_rows
  local target_rows
  if [[ -f "${DECK_STAGE_FILE}" ]]; then
    current_stage="$(cat "${DECK_STAGE_FILE}")"
  fi
  if (( AUTO_EXPAND_MAX_STAGE > 0 && current_stage >= AUTO_EXPAND_MAX_STAGE )); then
    printf '[%s] deck expansion skipped: max stage reached (%s)\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${current_stage}" | tee -a "${LOOP_LOG}"
    return 0
  fi
  local next_stage=$((current_stage + 1))
  current_rows="$(deck_rows)"
  target_rows=$((current_rows + STALL_DATASET_GROWTH_ROWS))
  printf '[%s] deck expansion triggered: stage=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${next_stage}" | tee -a "${LOOP_LOG}"
  if (( current_rows == 0 )); then
    python3 "${DECK_BUILDER}" >>"${LOOP_LOG}" 2>&1 || true
    current_rows="$(deck_rows)"
    target_rows=$((current_rows + STALL_DATASET_GROWTH_ROWS))
  fi
  python3 "${DECK_AUGMENTER}" --stage "${next_stage}" --target-rows "${target_rows}" >>"${LOOP_LOG}" 2>&1
  printf '%s\n' "${next_stage}" > "${DECK_STAGE_FILE}"
  printf '[%s] deck rows=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$(deck_rows)" | tee -a "${LOOP_LOG}"
  archive_search_state "good_deck_stage_${next_stage}"
}

adjust_model_once() {
  local current_stage=0
  if [[ -f "${MODEL_STAGE_FILE}" ]]; then
    current_stage="$(cat "${MODEL_STAGE_FILE}")"
  fi
  if (( current_stage >= AUTO_ADJUST_MODEL_MAX_STAGE )); then
    printf '[%s] model adjustment skipped: max stage reached (%s)\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${current_stage}" | tee -a "${LOOP_LOG}"
    apply_model_stage_overrides
    return 0
  fi
  local next_stage=$((current_stage + 1))
  printf '%s\n' "${next_stage}" > "${MODEL_STAGE_FILE}"
  printf '[%s] model adjustment triggered: stage=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${next_stage}" | tee -a "${LOOP_LOG}"
  apply_model_stage_overrides
  archive_search_state "good_model_stage_${next_stage}"
}

best_summary() {
  python3 - "${RESULTS_TSV}" <<'PY'
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8"), delimiter="\t"))
kept = [row for row in rows if row.get("status") == "keep"]
if not kept:
    print("[GOOD-CLEAN-SUMMARY] no kept runs yet")
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
    "[GOOD-CLEAN-SUMMARY] best={desc} | acc={acc:.4f} | f1={f1:.4f} | loss={loss:.6f}".format(
        desc=best.get("description", "-"),
        acc=float(best.get("val_accuracy") or "0"),
        f1=float(best.get("val_macro_f1") or "0"),
        loss=float(best.get("val_intent_loss") or "inf"),
    )
)
PY
}

llm_notify_summary() {
  local iteration="$1"
  INTENT_AUTORESEARCH_SUMMARY_PREFIX="good" INTENT_AUTORESEARCH_SUMMARY_LABEL="good-clean" python3 "${SUMMARY_SCRIPT}" "${iteration}" "${RESULTS_TSV}" "${ITERATION_LOG_DIR}" 2>>"${LOOP_LOG}" || return 1
}

send_discord() {
  local message="$1"
  python3 "${ROOT_DIR}/discord_notify.py" "${message}"
}

mark_best_only_recovery() {
  local reason="$1"
  python3 - "${RECOVERY_FLAG_FILE}" "${reason}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

payload = {
    "reason": sys.argv[2],
    "ts": datetime.now(timezone.utc).isoformat(),
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

iteration=0
write_status "starting"
while true; do
  apply_model_stage_overrides
  iteration=$((iteration + 1))
  if [[ "${EXPAND_EVERY_ITERATION}" == "1" ]]; then
    printf '[%s] deck expansion reason=scheduled-ambiguity iteration=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${iteration}" | tee -a "${LOOP_LOG}"
    expand_deck_once
  fi
  iteration_log="${ITERATION_LOG_DIR}/good_forever_iteration_${iteration}_$(date '+%Y%m%d_%H%M%S').log"
  status_line=""
  summary=""
  printf '\n[%s] good-forever iteration=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${iteration}" | tee -a "${LOOP_LOG}"
  printf '[%s] iteration_log=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${iteration_log}" | tee -a "${LOOP_LOG}"
  write_status "running"
  set +e
  if command -v timeout >/dev/null 2>&1; then
    timeout --signal=TERM "${SWEEP_TIMEOUT_SEC}s" bash "${SWEEP_SCRIPT}" >"${iteration_log}" 2>&1
  else
    bash "${SWEEP_SCRIPT}" >"${iteration_log}" 2>&1
  fi
  sweep_rc=$?
  set -e
  if [[ "${sweep_rc}" -ne 0 ]]; then
    if [[ "${sweep_rc}" -eq 124 ]]; then
      summary="sweep-timeout after ${SWEEP_TIMEOUT_SEC}s; restart from best on next boot"
    else
      summary="sweep-crash rc=${sweep_rc}; restart from best on next boot"
    fi
    printf '[%s] sweep failed rc=%s iteration=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${sweep_rc}" "${iteration}" | tee -a "${LOOP_LOG}"
    write_status "crash"
    if [[ "${RESET_TO_BEST_ON_CRASH}" == "1" ]]; then
      mark_best_only_recovery "good-sweep-crash-iter-${iteration}-rc-${sweep_rc}"
      printf '[%s] recovery flag written: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${RECOVERY_FLAG_FILE}" | tee -a "${LOOP_LOG}"
    fi
    exit "${sweep_rc}"
  fi
  rc=0
  status_line="$(python3 - "${RESULTS_TSV}" "${TARGET_ACCURACY}" "${TARGET_MACRO_F1}" <<'PY'
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8"), delimiter="\t"))
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
  printf '[%s] best=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${status_line}" | tee -a "${LOOP_LOG}"
  write_status "iteration-complete"
  current_json="$(current_best_json)"
  expand_reason="$(deck_expansion_signal "${current_json}")"
  should_expand=0
  if [[ "${EXPAND_EVERY_ITERATION}" != "1" ]]; then
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
  fi
  if [[ "${expand_reason}" == "stall" && "${AUTO_ADJUST_MODEL_ON_STALL}" == "1" ]]; then
    adjust_model_once
  fi
  if (( iteration % NOTIFY_EVERY == 0 )); then
    summary=""
    if ! summary="$(llm_notify_summary "${iteration}")"; then
      summary=""
    fi
    if [[ -z "${summary}" ]]; then
      summary="$(best_summary)"
    fi
    printf '[%s] notify=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${summary}" | tee -a "${LOOP_LOG}"
    write_status "notify-ready"
    status_text="$(status_message)"
    if send_discord "${status_text}" 2>>"${LOOP_LOG}"; then
      write_status "notified"
    else
      write_status "notify-failed"
      printf '[%s] discord notify failed\n' "$(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOOP_LOG}"
    fi
  fi
  if [[ "${rc}" -eq 10 ]]; then
    if can_expand_deck; then
      printf '[%s] threshold reached: advancing good-intent curriculum instead of stopping\n' "$(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOOP_LOG}"
      expand_deck_once
      write_status "curriculum-advanced"
      sleep "${SLEEP_SEC}"
      continue
    fi
    if [[ "${AUTO_ADJUST_MODEL_ON_STALL}" == "1" ]] && can_adjust_model; then
      printf '[%s] threshold reached: raising good-intent model stage instead of stopping\n' "$(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOOP_LOG}"
      adjust_model_once
      write_status "model-advanced"
      sleep "${SLEEP_SEC}"
      continue
    fi
    printf '[%s] stopping: clean threshold reached and no further escalation is available\n' "$(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOOP_LOG}"
    write_status "stopped"
    break
  fi
  sleep "${SLEEP_SEC}"
done
