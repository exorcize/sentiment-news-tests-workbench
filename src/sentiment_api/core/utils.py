import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax values for array."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def normalize_label(raw_label: str) -> str:
    """Normalize label to standard sentiment format."""
    normalized = raw_label.lower()
    if normalized in {"label_0", "negative"}:
        return "negative"
    if normalized in {"label_1", "neutral"}:
        return "neutral"
    if normalized in {"label_2", "positive"}:
        return "positive"
    return raw_label


def apply_sentiment_rules(
    text: str, rules: list[tuple[list[str], str]]
) -> tuple[str | None, bool]:
    """
    Apply override rules to sentiment prediction.

    Returns tuple of (overridden_label, was_overridden).
    """
    lower = text.lower()
    for phrases, sentiment_label in rules:
        if any(p in lower for p in phrases):
            return sentiment_label, True
    return None, False
