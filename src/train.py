"""Обучение классификатора тональности: TF-IDF + LogisticRegression.

Гиперпараметры подбираются по validation-выборке, test используется один раз
для итоговой оценки. В artifacts/metrics.json пишутся только метрики на test.
"""

import json
import pickle
import time
from itertools import product
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.utils import clean_text, compute_metrics

DATA_DIR = Path("data/raw")
ARTIFACTS_DIR = Path("artifacts")
SEED = 42

# Сетка гиперпараметров, перебираемая по validation.
NGRAM_RANGES = [(1, 1), (1, 2)]
MIN_DFS = [2, 5]
CS = [0.3, 1.0, 3.0, 10.0]


def load_split(name: str) -> pd.DataFrame:
    """Загружает сплит из data/raw и приводит тексты к виду для классификатора."""
    path = DATA_DIR / f"{name}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} не найден. Сначала выполните: uv run python -m src.get_data"
        )
    df = pd.read_json(path, lines=True)
    df["text"] = df["text"].map(clean_text)
    return df


def make_vectorizer(
    ngram_range: tuple[int, int], min_df: int, preprocessor=None
) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=ngram_range, min_df=min_df, sublinear_tf=True, preprocessor=preprocessor
    )


def make_classifier(c: float) -> LogisticRegression:
    return LogisticRegression(C=c, max_iter=1000, random_state=SEED)


def main() -> None:
    started = time.time()
    train, val, test = (load_split(s) for s in ("train", "validation", "test"))
    print(f"train={len(train)}, validation={len(val)}, test={len(test)}")
    print(f"Пример после предобработки: {train['text'].iloc[0][:150]!r}")

    # Подбор гиперпараметров: обучаем на train, сравниваем на validation.
    best_params, best_score = None, -1.0
    for ngram_range, min_df in product(NGRAM_RANGES, MIN_DFS):
        vectorizer = make_vectorizer(ngram_range, min_df)
        x_train = vectorizer.fit_transform(train["text"])
        x_val = vectorizer.transform(val["text"])
        for c in CS:
            clf = make_classifier(c).fit(x_train, train["label"])
            score = compute_metrics(val["label"], clf.predict(x_val))["f1_macro"]
            params = {"ngram_range": ngram_range, "min_df": min_df, "C": c}
            print(f"validation f1_macro={score:.4f}  {params}")
            if score > best_score:
                best_params, best_score = params, score
    print(f"Лучшие гиперпараметры: {best_params} (validation f1_macro={best_score:.4f})")

    # Итоговая модель: лучшие параметры, обучение на train. clean_text внутри пайплайна
    # (идемпотентен) позволяет подавать в сохранённую модель сырой текст.
    model = Pipeline(
        [
            (
                "tfidf",
                make_vectorizer(best_params["ngram_range"], best_params["min_df"], clean_text),
            ),
            ("clf", make_classifier(best_params["C"])),
        ]
    ).fit(train["text"], train["label"])

    # Единственная оценка на test.
    test_metrics = compute_metrics(test["label"], model.predict(test["text"]))

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    with open(ARTIFACTS_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(ARTIFACTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"Test metrics: {json.dumps(test_metrics)}")
    print(f"Сохранено: {ARTIFACTS_DIR / 'model.pkl'}, {ARTIFACTS_DIR / 'metrics.json'}")
    print(f"Время обучения: {time.time() - started:.0f} с")


if __name__ == "__main__":
    main()
