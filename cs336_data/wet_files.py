"""WET 语料的英文筛选规则。"""
from collections.abc import Callable
from pathlib import Path

import fasttext

from cs336_data.common import get_shared_assets_path


def load_english_filter(model_path: Path | None = None) -> Callable[[str], bool]:
    if model_path is None:
        model_path = get_shared_assets_path() / "classifiers/lid.176.bin"
    model = fasttext.load_model(str(model_path))

    def is_english(text: str) -> bool:
        text = text.replace("\n", " ").strip()
        if not text:
            return False
        labels, scores = model.predict(text)
        return labels[0] == "__label__en" and float(scores[0]) >= 0.7

    return is_english
