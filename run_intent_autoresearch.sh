#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${INTENT_AUTORESEARCH_LOG_DIR:-${ROOT_DIR}/logs}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${INTENT_AUTORESEARCH_RUN_NAME:-intentautoresearch_${RUN_TS}}"
TRAIN_LOG="${LOG_DIR}/${RUN_NAME}_train.log"
RESULTS_TSV="${INTENT_AUTORESEARCH_RESULTS_TSV:-${ROOT_DIR}/results.tsv}"
BEST_CONFIG_JSON="${INTENT_AUTORESEARCH_BEST_CONFIG_JSON:-${ROOT_DIR}/best_config.json}"
EXPERIMENT_JSON="${INTENT_AUTORESEARCH_EXPERIMENT_JSON:-${ROOT_DIR}/.last_experiment.json}"
DECK_MD="${INTENT_AUTORESEARCH_DECK_MD:-${ROOT_DIR}/reports/intent_research_deck.md}"
SELECTION_MODE="${INTENT_AUTORESEARCH_SELECTION_MODE:-loss}"
PLANNER_SCRIPT="${ROOT_DIR}/suggest_experiment.py"
DISCORD_NOTIFY_SCRIPT="${ROOT_DIR}/discord_notify.py"

mkdir -p "${LOG_DIR}"

ensure_results_header() {
  if [[ ! -f "${RESULTS_TSV}" ]]; then
    printf 'commit\tval_intent_loss\tval_accuracy\tval_macro_f1\tmemory_gb\tstatus\tdescription\n' > "${RESULTS_TSV}"
  fi
}

ensure_best_config() {
  python3 - "${BEST_CONFIG_JSON}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
defaults = {
    "description": "baseline",
    "INTENT_AUTORESEARCH_LR": "0.002",
    "INTENT_AUTORESEARCH_BATCH_SIZE": "16",
    "INTENT_AUTORESEARCH_EMBED_DIM": "128",
    "INTENT_AUTORESEARCH_HIDDEN_DIM": "192",
    "INTENT_AUTORESEARCH_DROPOUT": "0.1",
    "INTENT_AUTORESEARCH_WEIGHT_DECAY": "0.01",
    "INTENT_AUTORESEARCH_MAX_LEN": "48",
    "INTENT_AUTORESEARCH_TIME_BUDGET": "30",
}
current = {}
if path.exists():
    current = json.loads(path.read_text())
merged = dict(defaults)
merged.update(current)
path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
PY
}

choose_experiment() {
  if [[ -n "${INTENT_AUTORESEARCH_MANUAL_DESCRIPTION:-}" ]]; then
    python3 - "${BEST_CONFIG_JSON}" "${INTENT_AUTORESEARCH_MANUAL_DESCRIPTION}" <<'PY' > "${EXPERIMENT_JSON}"
import json
import sys
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text())
cfg["description"] = sys.argv[2]
print(json.dumps(cfg, indent=2))
PY
    return
  fi
  if [[ "${INTENT_AUTORESEARCH_USE_LLM_PLANNER:-${INTENT_AUTORESEARCH_USE_CODEX_PLANNER:-1}}" != "0" ]]; then
    if python3 "${PLANNER_SCRIPT}" "${BEST_CONFIG_JSON}" "${RESULTS_TSV}" > "${EXPERIMENT_JSON}"; then
      return
    fi
  fi
  python3 - "${BEST_CONFIG_JSON}" "${RESULTS_TSV}" <<'PY' > "${EXPERIMENT_JSON}"
import csv
import json
import sys
from pathlib import Path

best = json.loads(Path(sys.argv[1]).read_text())
results_path = Path(sys.argv[2])
rows = []
if results_path.exists():
    with results_path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
run_idx = len(rows)

def clone(desc, **updates):
    cfg = dict(best)
    cfg.update(updates)
    cfg["description"] = desc
    return cfg

candidates = [
    clone("baseline-repeat"),
    clone("lower-lr", INTENT_AUTORESEARCH_LR="0.001"),
    clone("higher-lr", INTENT_AUTORESEARCH_LR="0.004"),
    clone("smaller-batch", INTENT_AUTORESEARCH_BATCH_SIZE="8"),
    clone("larger-batch", INTENT_AUTORESEARCH_BATCH_SIZE="24"),
    clone("wider-hidden", INTENT_AUTORESEARCH_HIDDEN_DIM="256"),
    clone("smaller-hidden", INTENT_AUTORESEARCH_HIDDEN_DIM="128"),
    clone("larger-embed", INTENT_AUTORESEARCH_EMBED_DIM="192"),
    clone("lower-dropout", INTENT_AUTORESEARCH_DROPOUT="0.05"),
    clone("higher-dropout", INTENT_AUTORESEARCH_DROPOUT="0.2"),
    clone("stronger-weight-decay", INTENT_AUTORESEARCH_WEIGHT_DECAY="0.03"),
    clone("low-lr-strong-reg", INTENT_AUTORESEARCH_LR="0.0005", INTENT_AUTORESEARCH_DROPOUT="0.2", INTENT_AUTORESEARCH_WEIGHT_DECAY="0.03"),
    clone("shorter-context", INTENT_AUTORESEARCH_MAX_LEN="32"),
    clone("longer-context", INTENT_AUTORESEARCH_MAX_LEN="64"),
]
print(json.dumps(candidates[run_idx % len(candidates)], indent=2))
PY
}

read_config_field() {
  local field="$1"
  python3 - "${EXPERIMENT_JSON}" "${field}" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
print(data[sys.argv[2]])
PY
}

read_config_field_optional() {
  local field="$1"
  python3 - "${EXPERIMENT_JSON}" "${field}" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1]))
print(data.get(sys.argv[2], ""))
PY
}

resolve_env_value() {
  local key="$1"
  local configured
  configured="$(read_config_field "${key}")"
  local current="${!key-}"
  if [[ -n "${current}" ]]; then
    printf '%s\n' "${current}"
  else
    printf '%s\n' "${configured}"
  fi
}

build_run_report() {
  local description="$1"
  local planner_source="$2"
  local planner_note="$3"
  local status="$4"
  local val_loss="$5"
  local val_acc="$6"
  local val_f1="$7"
  local memory_gb="$8"
  python3 - "${description}" "${planner_source}" "${planner_note}" "${status}" "${val_loss}" "${val_acc}" "${val_f1}" "${memory_gb}" "${DECK_MD}" "${TRAIN_LOG}" <<'PY'
import sys
from pathlib import Path

description, planner_source, planner_note, status, val_loss, val_acc, val_f1, memory_gb, deck_md, train_log = sys.argv[1:]
planner = planner_source or "heuristic"
note = f" | note={planner_note}" if planner_note else ""
message = (
    f"[intentautoresearch] status={status} | exp={description} | planner={planner}"
    f" | loss={float(val_loss):.6f} | acc={float(val_acc):.4f} | f1={float(val_f1):.4f}"
    f" | vram_gb={float(memory_gb):.1f}{note}"
    f" | deck={Path(deck_md).name} | log={Path(train_log).name}"
)
print(message[:1900])
PY
}

maybe_send_discord_report() {
  local message="$1"
  if [[ "${INTENT_AUTORESEARCH_SEND_DISCORD_REPORT:-1}" == "0" ]]; then
    return 0
  fi
  if python3 "${DISCORD_NOTIFY_SCRIPT}" "${message}"; then
    echo "Discord report sent"
  else
    echo "Discord report failed" >&2
  fi
}

prepare_data() {
  echo "Preparing intent dataset"
  cd "${ROOT_DIR}"
  python3 prepare.py
}

run_training() {
  cd "${ROOT_DIR}"
  ensure_results_header
  ensure_best_config
  choose_experiment
  local description
  local planner_source
  local planner_note
  description="$(read_config_field description)"
  planner_source="$(read_config_field_optional _planner_source)"
  planner_note="$(read_config_field_optional _planner_note)"
  echo "Selected experiment: ${description}"
  echo "Planner source: ${planner_source:-heuristic}"
  if [[ -n "${planner_note}" ]]; then
    echo "Planner note: ${planner_note}"
  fi
  set +e
  env \
    INTENT_AUTORESEARCH_LR="$(resolve_env_value INTENT_AUTORESEARCH_LR)" \
    INTENT_AUTORESEARCH_BATCH_SIZE="$(resolve_env_value INTENT_AUTORESEARCH_BATCH_SIZE)" \
    INTENT_AUTORESEARCH_EMBED_DIM="$(resolve_env_value INTENT_AUTORESEARCH_EMBED_DIM)" \
    INTENT_AUTORESEARCH_HIDDEN_DIM="$(resolve_env_value INTENT_AUTORESEARCH_HIDDEN_DIM)" \
    INTENT_AUTORESEARCH_DROPOUT="$(resolve_env_value INTENT_AUTORESEARCH_DROPOUT)" \
    INTENT_AUTORESEARCH_WEIGHT_DECAY="$(resolve_env_value INTENT_AUTORESEARCH_WEIGHT_DECAY)" \
    INTENT_AUTORESEARCH_MAX_LEN="$(resolve_env_value INTENT_AUTORESEARCH_MAX_LEN)" \
    INTENT_AUTORESEARCH_TIME_BUDGET="$(resolve_env_value INTENT_AUTORESEARCH_TIME_BUDGET)" \
    python3 train.py | tee "${TRAIN_LOG}"
  local train_status=${PIPESTATUS[0]}
  set -e
  log_result "${description}" "${planner_source:-heuristic}" "${planner_note}" "${train_status}"
}

extract_metric() {
  local key="$1"
  local log_file="$2"
  python3 - "${key}" "${log_file}" <<'PY'
import re, sys
key = sys.argv[1]
text = open(sys.argv[2], encoding="utf-8").read()
match = re.search(rf"^{re.escape(key)}:\s+([0-9.]+)$", text, re.MULTILINE)
print(match.group(1) if match else "")
PY
}

best_score() {
  python3 - "${RESULTS_TSV}" "${SELECTION_MODE}" <<'PY'
import csv
import math
import sys

mode = sys.argv[2]
rows = []
with open(sys.argv[1], newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))

kept = [row for row in rows if row.get("status") == "keep"]
if not kept:
    print("")
    raise SystemExit(0)

def score(row):
    loss = float(row.get("val_intent_loss") or "inf")
    acc = float(row.get("val_accuracy") or "0")
    f1 = float(row.get("val_macro_f1") or "0")
    if mode == "macro_f1":
        return (f1, acc, -loss)
    return (-loss,)

best = max(kept, key=score)
if mode == "macro_f1":
    print(f"{float(best['val_macro_f1']):.4f}\t{float(best['val_accuracy']):.4f}\t{float(best['val_intent_loss']):.6f}")
else:
    print(f"{float(best['val_intent_loss']):.6f}")
PY
}

append_result_row() {
  local commit_hash="$1"
  local loss="$2"
  local acc="$3"
  local f1="$4"
  local memory_gb="$5"
  local status="$6"
  local description="$7"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${commit_hash}" "${loss}" "${acc}" "${f1}" "${memory_gb}" "${status}" "${description}" >> "${RESULTS_TSV}"
}

update_best_config() {
  local description="$1"
  python3 - "${BEST_CONFIG_JSON}" "${description}" \
    "$(resolve_env_value INTENT_AUTORESEARCH_LR)" \
    "$(resolve_env_value INTENT_AUTORESEARCH_BATCH_SIZE)" \
    "$(resolve_env_value INTENT_AUTORESEARCH_EMBED_DIM)" \
    "$(resolve_env_value INTENT_AUTORESEARCH_HIDDEN_DIM)" \
    "$(resolve_env_value INTENT_AUTORESEARCH_DROPOUT)" \
    "$(resolve_env_value INTENT_AUTORESEARCH_WEIGHT_DECAY)" \
    "$(resolve_env_value INTENT_AUTORESEARCH_MAX_LEN)" \
    "$(resolve_env_value INTENT_AUTORESEARCH_TIME_BUDGET)" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "description": sys.argv[2],
    "INTENT_AUTORESEARCH_LR": sys.argv[3],
    "INTENT_AUTORESEARCH_BATCH_SIZE": sys.argv[4],
    "INTENT_AUTORESEARCH_EMBED_DIM": sys.argv[5],
    "INTENT_AUTORESEARCH_HIDDEN_DIM": sys.argv[6],
    "INTENT_AUTORESEARCH_DROPOUT": sys.argv[7],
    "INTENT_AUTORESEARCH_WEIGHT_DECAY": sys.argv[8],
    "INTENT_AUTORESEARCH_MAX_LEN": sys.argv[9],
    "INTENT_AUTORESEARCH_TIME_BUDGET": sys.argv[10],
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

log_result() {
  local description="$1"
  local planner_source="$2"
  local planner_note="$3"
  local train_status="$4"
  local val_loss val_acc val_f1 peak_vram_mb memory_gb status commit_hash best_so_far
  val_loss="$(extract_metric val_intent_loss "${TRAIN_LOG}")"
  val_acc="$(extract_metric val_accuracy "${TRAIN_LOG}")"
  val_f1="$(extract_metric val_macro_f1 "${TRAIN_LOG}")"
  peak_vram_mb="$(extract_metric peak_vram_mb "${TRAIN_LOG}")"
  commit_hash="$(git -C "${ROOT_DIR}" rev-parse --short HEAD 2>/dev/null || echo nogit)"
  if [[ "${train_status}" -ne 0 || -z "${val_loss}" || -z "${val_acc}" || -z "${val_f1}" ]]; then
    append_result_row "${commit_hash}" "0.000000" "0.0000" "0.0000" "0.0" "crash" "${description}"
    INTENT_AUTORESEARCH_RESULTS_TSV="${RESULTS_TSV}" \
      INTENT_AUTORESEARCH_BEST_CONFIG_JSON="${BEST_CONFIG_JSON}" \
      INTENT_AUTORESEARCH_DECK_MD="${DECK_MD}" \
      python3 generate_deck.py >/dev/null
    maybe_send_discord_report "$(build_run_report "${description}" "${planner_source}" "${planner_note}" "crash" "0.000000" "0.0000" "0.0000" "0.0")"
    echo "Experiment status: crash"
    return 1
  fi
  memory_gb="$(python3 - "${peak_vram_mb:-0}" <<'PY'
import sys
print(f"{float(sys.argv[1]) / 1024.0:.1f}")
PY
)"
  best_so_far="$(best_score)"
  status="keep"
  if [[ -n "${best_so_far}" ]]; then
    status="$(python3 - "${SELECTION_MODE}" "${val_loss}" "${val_acc}" "${val_f1}" "${best_so_far}" <<'PY'
import sys
mode = sys.argv[1]
loss = float(sys.argv[2])
acc = float(sys.argv[3])
f1 = float(sys.argv[4])
best = sys.argv[5]
if mode == "macro_f1":
    best_f1, best_acc, best_loss = [float(x) for x in best.split("\t")]
    candidate = (f1, acc, -loss)
    champion = (best_f1, best_acc, -best_loss)
    print("keep" if candidate > champion else "discard")
else:
    champion_loss = float(best)
    print("keep" if loss < champion_loss else "discard")
PY
)"
  fi
  append_result_row "${commit_hash}" "${val_loss}" "${val_acc}" "${val_f1}" "${memory_gb}" "${status}" "${description}"
  if [[ "${status}" == "keep" ]]; then
    update_best_config "${description}"
  fi
  INTENT_AUTORESEARCH_RESULTS_TSV="${RESULTS_TSV}" \
    INTENT_AUTORESEARCH_BEST_CONFIG_JSON="${BEST_CONFIG_JSON}" \
    INTENT_AUTORESEARCH_DECK_MD="${DECK_MD}" \
    python3 generate_deck.py >/dev/null
  maybe_send_discord_report "$(build_run_report "${description}" "${planner_source}" "${planner_note}" "${status}" "${val_loss}" "${val_acc}" "${val_f1}" "${memory_gb}")"
  echo "Experiment status: ${status} (selection_mode=${SELECTION_MODE}, val_intent_loss=${val_loss}, best_before=${best_so_far:-none})"
}

prepare_data
run_training
echo "Train log: ${TRAIN_LOG}"
echo "Deck: ${DECK_MD}"
