#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${ROOT_DIR}/run_intent_autoresearch.sh"
DATA_FILE="${ROOT_DIR}/data/good_intent_clean_deck.jsonl"
RESULTS_TSV="${ROOT_DIR}/good_results.tsv"
BEST_CONFIG_JSON="${ROOT_DIR}/good_best_config.json"
DECK_MD="${ROOT_DIR}/reports/good_intent_research_deck.md"
DECK_BUILDER="${ROOT_DIR}/build_good_intent_deck.py"
ARTIFACT_DIR="${ROOT_DIR}/artifacts_good"
DEVICE="${INTENT_AUTORESEARCH_DEVICE:-cpu}"

if [[ ! -f "${DATA_FILE}" ]]; then
  python3 "${DECK_BUILDER}"
fi

if [[ ! -f "${DATA_FILE}" ]]; then
  echo "Missing good intent deck: ${DATA_FILE}" >&2
  exit 1
fi

if [[ "${INTENT_AUTORESEARCH_USE_LLM_PLANNER:-${INTENT_AUTORESEARCH_USE_CODEX_PLANNER:-1}}" == "1" ]]; then
  echo "=== prometheus-good-planner ==="
  env \
    INTENT_AUTORESEARCH_DISCORD_ROUTE="good" \
    INTENT_AUTORESEARCH_DATA_FILES="${DATA_FILE}" \
    INTENT_AUTORESEARCH_RESULTS_TSV="${RESULTS_TSV}" \
    INTENT_AUTORESEARCH_BEST_CONFIG_JSON="${BEST_CONFIG_JSON}" \
    INTENT_AUTORESEARCH_DECK_MD="${DECK_MD}" \
    INTENT_AUTORESEARCH_ARTIFACT_DIR="${ARTIFACT_DIR}" \
    INTENT_AUTORESEARCH_DEVICE="${DEVICE}" \
    INTENT_AUTORESEARCH_SELECTION_MODE="macro_f1" \
    "${RUNNER}"
  echo "Good results: ${RESULTS_TSV}"
  echo "Good deck: ${DECK_MD}"
  exit 0
fi

echo "=== heuristic-good-planner ==="
env \
  INTENT_AUTORESEARCH_DISCORD_ROUTE="good" \
  INTENT_AUTORESEARCH_DATA_FILES="${DATA_FILE}" \
  INTENT_AUTORESEARCH_RESULTS_TSV="${RESULTS_TSV}" \
  INTENT_AUTORESEARCH_BEST_CONFIG_JSON="${BEST_CONFIG_JSON}" \
  INTENT_AUTORESEARCH_DECK_MD="${DECK_MD}" \
  INTENT_AUTORESEARCH_ARTIFACT_DIR="${ARTIFACT_DIR}" \
  INTENT_AUTORESEARCH_DEVICE="${DEVICE}" \
  INTENT_AUTORESEARCH_SELECTION_MODE="macro_f1" \
  "${RUNNER}"

echo "Good results: ${RESULTS_TSV}"
echo "Good deck: ${DECK_MD}"
