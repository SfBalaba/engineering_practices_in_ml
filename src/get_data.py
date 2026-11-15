"""Загрузка датасета Kinopoisk sentiment с Hugging Face Hub в data/raw/.

Токен (нужен только для приватных датасетов или повышения лимитов) читается
из переменной окружения HF_TOKEN и нигде не хранится в репозитории.
"""

import os
import shutil
from pathlib import Path

from datasets import load_dataset

DATASET = "ai-forever/kinopoisk-sentiment-classification"
OUTPUT_DIR = Path("data/raw")
SPLITS = ("train", "validation", "test")
COLUMNS = ("text", "label")


def main() -> None:
    # Каждый запуск загружает данные с нуля, без использования старого кэша.
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    token = os.environ.get("HF_TOKEN") or None
    dataset = load_dataset(DATASET, token=token, download_mode="force_redownload")

    for split in SPLITS:
        if split not in dataset:
            raise RuntimeError(f"В датасете {DATASET} нет сплита '{split}'")
        part = dataset[split].select_columns(list(COLUMNS))
        part.to_json(OUTPUT_DIR / f"{split}.jsonl", lines=True, force_ascii=False)

    # Проверка, что все ожидаемые файлы на месте и не пустые.
    for split in SPLITS:
        path = OUTPUT_DIR / f"{split}.jsonl"
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Ожидаемый файл {path} не создан или пуст")
        print(f"{path}: {len(dataset[split])} строк, {path.stat().st_size} байт")

    print(f"Данные сохранены в {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
