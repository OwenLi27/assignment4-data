from pathlib import Path

import fasttext

from cs336_data.data_preparation import normalize_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    PROJECT_ROOT
    / "local-shared-data/quality/experiments-expanded/models"
    / "6f8b29d19296ca3f05af.bin"
)

model = fasttext.load_model(str(MODEL_PATH))


def classify_quality(text: str) -> tuple[str, float]:
    text = normalize_text(text)
    if not text:
        raise ValueError("the content to be classified is empty after normalization")
    # 与训练数据使用相同的截断规则
    if len(text) > 50000:
        text = text[:50000].rsplit(' ', 1)[0]
    labels, scores = model.predict(text)
    confidence = max(0.0, min(1.0, float(scores[0])))
    return labels[0].removeprefix('__label__'), confidence
