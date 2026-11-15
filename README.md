# Инженерные практики в ML

## Задача

Классификация тональности русскоязычных отзывов (sentiment analysis, 3 класса:
0 — негативный, 1 — нейтральный, 2 — позитивный).
Модель: TF-IDF + логистическая регрессия (scikit-learn).

## Датасет

- Источник: <https://huggingface.co/datasets/ai-forever/kinopoisk-sentiment-classification>
- Лицензия: MIT
- Сплиты: train (10 500), validation (1 500), test (1 500), классы сбалансированы.

Данные сохраняются в `data/raw/{train,validation,test}.jsonl` (в Git не хранятся).

## Установка

Требуется [uv](https://docs.astral.sh/uv/) (Python 3.13 подтянется автоматически).
Зависимости ставятся строго из `uv.lock`:

```bash
uv sync --locked
uv run pre-commit install   # опционально: хуки при git commit
```

## Загрузка данных

```bash
uv run python -m src.get_data
```

Скрипт загружает датасет с нуля (пересоздаёт `data/raw/`) и проверяет, что все
ожидаемые файлы созданы и не пусты. Датасет публичный, токен не нужен. Если требуется
токен Hugging Face, задайте его через переменную окружения —
в Git он не хранится:

```bash
export HF_TOKEN=hf_...
```

## Проверка кода

```bash
uv run ruff check .
uv run ruff format --check .
uv run pre-commit run --all-files
```

Настройки Ruff — в `pyproject.toml`, хуки (ruff, ruff-format, проверка `uv.lock`,
базовые проверки файлов) — в `.pre-commit-config.yaml`.

## Обучение

```bash
uv run python -m src.train
```

Обучение занимает ~3 минуты на CPU.

- Предобработка текста (`src/utils.py::clean_text`): нижний регистр, `ё → е`, удаление
  ссылок и HTML-тегов, замена пунктуации на пробелы.
- Подбор гиперпараметров (`ngram_range`, `min_df`, `C`) идёт по **validation**.
- Лучшая модель обучается на train и один раз оценивается на **test**.
- Сохраняются (заново при каждом запуске):
  - `artifacts/model.pkl` — pickle `sklearn.Pipeline` (TF-IDF + LogisticRegression);
  - `artifacts/metrics.json` — только метрики на test (`accuracy`, `f1_macro`).

Использование модели (из корня репозитория, пайплайн принимает сырой текст):

```python
import pickle

with open("artifacts/model.pkl", "rb") as f:
    model = pickle.load(f)
print(model.predict(["Отличный фильм!"]))  # 0 — негатив, 1 — нейтрально, 2 — позитив
```

> `pickle.load` выполняет произвольный код — загружайте только собственные артефакты.

> `src/train_bert.py` — отложенный эксперимент с `bert-base-uncased`; для его запуска нужны
> `torch` и `transformers`, которых в зависимостях проекта нет.

## Отчёт

Результаты запусков: [reports/hw-1.md](reports/hw-1.md).
