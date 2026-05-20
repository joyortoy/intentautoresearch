"""
Prepare normalized intent-classification artifacts.
"""

from __future__ import annotations

import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = Path(os.getenv("INTENT_AUTORESEARCH_ARTIFACT_DIR", str(ROOT / "artifacts"))).expanduser()
DATA_JSON = ARTIFACT_DIR / "dataset.json"
META_JSON = ARTIFACT_DIR / "metadata.json"
DEFAULT_SEED = 42
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
TOKEN_RE = re.compile(r"[A-Za-z0-9_:/.-]+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class IntentExample:
    text: str
    intent: str
    source: str


def _extract_instruction_text(instruction: str) -> str:
    match = re.search(r"'([^']+)'", instruction)
    if match:
        return match.group(1).strip()
    return instruction.strip()


def normalize_record(record: dict, source: str) -> IntentExample | None:
    text = ""
    if isinstance(record.get("prompt"), str):
        text = record["prompt"].strip()
    elif isinstance(record.get("text"), str):
        text = record["text"].strip()
    elif isinstance(record.get("instruction"), str):
        text = _extract_instruction_text(record["instruction"])
    intent = ""
    for key in ("expected_intent", "intent", "output", "label"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            intent = value.strip()
            break
    if not text or not intent:
        return None
    return IntentExample(text=text, intent=intent, source=source)


def iter_examples(path: Path) -> Iterable[IntentExample]:
    with path.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            example = normalize_record(record, source=f"{path}:{idx}")
            if example is not None:
                yield example


def default_data_files() -> list[Path]:
    env = os.getenv("INTENT_AUTORESEARCH_DATA_FILES", "").strip()
    if env:
        return [Path(part).expanduser() for part in env.split(":") if part.strip()]
    candidates = [
        Path("/home/sam/memory/intentstack/intent_eval_set.jsonl"),
        Path("/home/sam/memory/intentstack/intent_eval_confusing_set.jsonl"),
        ROOT / "data" / "seed_intent_examples.jsonl",
    ]
    return [path for path in candidates if path.exists()]


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build_vocab(texts: Iterable[str], min_freq: int = 1) -> list[str]:
    counts = Counter()
    for text in texts:
        counts.update(tokenize(text))
    vocab = [PAD_TOKEN, UNK_TOKEN]
    vocab.extend(sorted(token for token, count in counts.items() if count >= min_freq))
    return vocab


def stratified_split(examples: list[IntentExample], seed: int = DEFAULT_SEED) -> tuple[list[IntentExample], list[IntentExample]]:
    rng = random.Random(seed)
    by_intent: dict[str, list[IntentExample]] = defaultdict(list)
    for example in examples:
        by_intent[example.intent].append(example)
    train: list[IntentExample] = []
    val: list[IntentExample] = []
    for intent, group in sorted(by_intent.items()):
        rng.shuffle(group)
        if len(group) == 1:
            train.extend(group)
            continue
        val_count = max(1, round(len(group) * 0.2))
        if val_count >= len(group):
            val_count = len(group) - 1
        val.extend(group[:val_count])
        train.extend(group[val_count:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def serialize_examples(examples: list[IntentExample]) -> list[dict]:
    return [{"text": ex.text, "intent": ex.intent, "source": ex.source} for ex in examples]


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    files = default_data_files()
    if not files:
        raise SystemExit("No dataset files found. Set INTENT_AUTORESEARCH_DATA_FILES or add data/seed_intent_examples.jsonl.")
    dedup: dict[tuple[str, str], IntentExample] = {}
    for path in files:
        for example in iter_examples(path):
            dedup[(example.text, example.intent)] = example
    examples = sorted(dedup.values(), key=lambda ex: (ex.intent, ex.text))
    if len(examples) < 8:
        raise SystemExit(f"Need at least 8 normalized examples, found {len(examples)}.")
    train_examples, val_examples = stratified_split(examples)
    intents = sorted({ex.intent for ex in examples})
    vocab = build_vocab(ex.text for ex in train_examples)
    metadata = {
        "num_examples": len(examples),
        "num_train": len(train_examples),
        "num_val": len(val_examples),
        "num_intents": len(intents),
        "intent_names": intents,
        "vocab_size": len(vocab),
        "vocab": vocab,
        "data_files": [str(path) for path in files],
        "seed": DEFAULT_SEED,
    }
    dataset = {
        "train": serialize_examples(train_examples),
        "val": serialize_examples(val_examples),
    }
    DATA_JSON.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    META_JSON.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared {len(examples)} examples")
    print(f"Train: {len(train_examples)} | Val: {len(val_examples)}")
    print(f"Intents: {len(intents)} | Vocab: {len(vocab)}")
    print(f"Artifacts: {ARTIFACT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
