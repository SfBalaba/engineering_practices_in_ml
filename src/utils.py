import re

from sklearn.metrics import accuracy_score, f1_score

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_NON_TEXT_RE = re.compile(r"[^a-zа-я0-9\s]")
_SPACES_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Предобработка отзыва для классификатора.

    Нижний регистр, ё -> е, удаление ссылок и HTML-тегов, замена пунктуации
    и прочих символов на пробел, схлопывание пробелов.
    """
    text = str(text).lower().replace("ё", "е")
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _NON_TEXT_RE.sub(" ", text)
    return _SPACES_RE.sub(" ", text).strip()


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Метрики классификации: accuracy и macro-F1."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }
