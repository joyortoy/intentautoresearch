"""
Pure intent-classification training loop with cross-entropy objective.
"""

from __future__ import annotations

import gc
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"
DATA_JSON = ARTIFACT_DIR / "dataset.json"
META_JSON = ARTIFACT_DIR / "metadata.json"
TIME_BUDGET = int(os.getenv("INTENT_AUTORESEARCH_TIME_BUDGET", "300"))
SEED = int(os.getenv("INTENT_AUTORESEARCH_SEED", "42"))
MAX_LEN = int(os.getenv("INTENT_AUTORESEARCH_MAX_LEN", "48"))
BATCH_SIZE = int(os.getenv("INTENT_AUTORESEARCH_BATCH_SIZE", "16"))
LR = float(os.getenv("INTENT_AUTORESEARCH_LR", "2e-3"))
WEIGHT_DECAY = float(os.getenv("INTENT_AUTORESEARCH_WEIGHT_DECAY", "0.01"))
EMBED_DIM = int(os.getenv("INTENT_AUTORESEARCH_EMBED_DIM", "128"))
HIDDEN_DIM = int(os.getenv("INTENT_AUTORESEARCH_HIDDEN_DIM", "192"))
DROPOUT = float(os.getenv("INTENT_AUTORESEARCH_DROPOUT", "0.1"))
DEVICE_REQUEST = os.getenv("INTENT_AUTORESEARCH_DEVICE", "auto").strip().lower()


def tokenize(text: str) -> list[str]:
    import re
    return re.findall(r"[A-Za-z0-9_:/.-]+|[^\w\s]", text.lower(), re.UNICODE)


def load_artifacts() -> tuple[dict, dict]:
    if not DATA_JSON.exists() or not META_JSON.exists():
        raise SystemExit("Artifacts missing. Run `uv run prepare.py` first.")
    dataset = json.loads(DATA_JSON.read_text(encoding="utf-8"))
    metadata = json.loads(META_JSON.read_text(encoding="utf-8"))
    return dataset, metadata


class IntentClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.proj = nn.Linear(EMBED_DIM, HIDDEN_DIM)
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.dropout = nn.Dropout(DROPOUT)
        self.head = nn.Linear(HIDDEN_DIM, num_classes)

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / denom
        pooled = self.dropout(F.gelu(self.norm(self.proj(pooled))))
        return self.head(pooled)


def encode_batch(examples: list[dict], token_to_id: dict[str, int], intent_to_id: dict[str, int], device: torch.device):
    ids = torch.zeros((len(examples), MAX_LEN), dtype=torch.long)
    mask = torch.zeros((len(examples), MAX_LEN), dtype=torch.float32)
    labels = torch.zeros((len(examples),), dtype=torch.long)
    unk = token_to_id["<unk>"]
    for row_idx, example in enumerate(examples):
        tokens = tokenize(example["text"])[:MAX_LEN]
        token_ids = [token_to_id.get(tok, unk) for tok in tokens]
        ids[row_idx, :len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
        mask[row_idx, :len(token_ids)] = 1.0
        labels[row_idx] = intent_to_id[example["intent"]]
    return ids.to(device), mask.to(device), labels.to(device)


def iterate_minibatches(examples: list[dict], batch_size: int):
    indices = list(range(len(examples)))
    random.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [examples[i] for i in indices[start:start + batch_size]]


@torch.no_grad()
def evaluate(model: IntentClassifier, examples: list[dict], token_to_id: dict[str, int], intent_to_id: dict[str, int], device: torch.device):
    model.eval()
    losses = []
    gold = []
    pred = []
    for batch in iterate_minibatches(examples, BATCH_SIZE):
        token_ids, mask, labels = encode_batch(batch, token_to_id, intent_to_id, device)
        logits = model(token_ids, mask)
        losses.append(F.cross_entropy(logits, labels).item())
        gold.extend(labels.cpu().tolist())
        pred.extend(logits.argmax(dim=-1).cpu().tolist())
    accuracy = sum(int(a == b) for a, b in zip(gold, pred)) / max(1, len(gold))
    by_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    classes = sorted(intent_to_id.values())
    for cls in classes:
        for g, p in zip(gold, pred):
            if g == cls and p == cls:
                by_class[cls]["tp"] += 1
            elif g != cls and p == cls:
                by_class[cls]["fp"] += 1
            elif g == cls and p != cls:
                by_class[cls]["fn"] += 1
    f1s = []
    for cls in classes:
        tp = by_class[cls]["tp"]
        fp = by_class[cls]["fp"]
        fn = by_class[cls]["fn"]
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))
    return {
        "val_intent_loss": sum(losses) / max(1, len(losses)),
        "val_accuracy": accuracy,
        "val_macro_f1": sum(f1s) / max(1, len(f1s)),
    }


def main() -> int:
    dataset, metadata = load_artifacts()
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    if DEVICE_REQUEST == "cuda":
        device = torch.device("cuda")
    elif DEVICE_REQUEST == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = metadata["vocab"]
    intent_names = metadata["intent_names"]
    token_to_id = {token: idx for idx, token in enumerate(vocab)}
    intent_to_id = {intent: idx for idx, intent in enumerate(intent_names)}

    model = IntentClassifier(vocab_size=len(vocab), num_classes=len(intent_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_examples = dataset["train"]
    val_examples = dataset["val"]
    print(f"Device: {device}")
    print(f"Train examples: {len(train_examples)} | Val examples: {len(val_examples)}")
    print(f"Intents: {len(intent_names)} | Vocab size: {len(vocab)}")
    print(f"Time budget: {TIME_BUDGET}s")

    start_wall = time.time()
    train_seconds = 0.0
    step = 0
    ema_loss = 0.0

    while True:
        batch_start = time.time()
        for batch in iterate_minibatches(train_examples, BATCH_SIZE):
            token_ids, mask, labels = encode_batch(batch, token_to_id, intent_to_id, device)
            logits = model(token_ids, mask)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_f = loss.item()
            ema_loss = 0.9 * ema_loss + 0.1 * loss_f if step else loss_f
            elapsed = time.time() - batch_start
            if step > 5:
                train_seconds += elapsed
            print(
                f"\rstep {step:05d} | train_loss: {ema_loss:.6f} | "
                f"elapsed: {train_seconds:.1f}s / {TIME_BUDGET}s",
                end="",
                flush=True,
            )
            batch_start = time.time()
            step += 1
            if step > 5 and train_seconds >= TIME_BUDGET:
                break
        if step > 5 and train_seconds >= TIME_BUDGET:
            break

    print()
    metrics = evaluate(model, val_examples, token_to_id, intent_to_id, device)
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0.0
    total_params = sum(p.numel() for p in model.parameters())
    gc.collect()

    print("---")
    print(f"val_intent_loss:  {metrics['val_intent_loss']:.6f}")
    print(f"val_accuracy:     {metrics['val_accuracy']:.4f}")
    print(f"val_macro_f1:     {metrics['val_macro_f1']:.4f}")
    print(f"training_seconds: {train_seconds:.1f}")
    print(f"total_seconds:    {time.time() - start_wall:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {total_params / 1e6:.2f}")
    print(f"num_classes:      {len(intent_names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
