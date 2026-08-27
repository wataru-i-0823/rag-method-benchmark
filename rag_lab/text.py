from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

TOKEN = re.compile(r"[A-Za-z0-9_]+|[\u3040-\u30ff\u3400-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """Tokenize English words and Japanese character n-grams without dependencies."""
    parts = TOKEN.findall(text.lower())
    tokens: list[str] = []
    for part in parts:
        tokens.append(part)
        if any("\u3040" <= char <= "\u9fff" for char in part):
            compact = part.replace(" ", "")
            tokens.extend(compact[i : i + 2] for i in range(max(0, len(compact) - 1)))
    return tokens


def cosine(left: Counter[str], right: Counter[str]) -> float:
    numerator = sum(value * right.get(key, 0) for key, value in left.items())
    if not numerator:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def normalise(scores: Iterable[float]) -> list[float]:
    values = list(scores)
    if not values or max(values) == min(values):
        return [1.0 if values else 0.0 for _ in values]
    low, high = min(values), max(values)
    return [(value - low) / (high - low) for value in values]
