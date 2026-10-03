import json
from pathlib import Path

from datasets import load_dataset

from markov_chain import Markovka
from tokenizer import RegexTokenizer

# Лимит задаёт количество записей, а не токенов. None — весь источник.
SOURCES = [
    ("Mikimi/russian-wikipedia-top100k", ("summary",), None),
    ("inkoziev/ru_stories", tuple(f"sentence{i}" for i in range(1, 6)), 2000),
    ("IlyaGusev/gazeta", ("text",), 2000),
    # # ("IlyaGusev/pikabu", ("text_markdown",), 2000),
    # ("IlyaGusev/ficbook", ("parts",), 10),
]

CACHE_FILE = Path(__file__).resolve().parent / "data" / "texts.json"
CACHE_VERSION = 1


def load_corpus() -> list[str]:
    sources = []
    for dataset_name, fields, limit in SOURCES:
        sources.append([dataset_name, list(fields), limit])

    if CACHE_FILE.exists():
        try:
            cached = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            print("Кеш повреждён, загружаем тексты заново...")
        else:
            if (
                isinstance(cached, dict)
                and cached.get("version") == CACHE_VERSION
                and cached.get("sources") == sources
                and isinstance(cached.get("texts"), list)
                and all(isinstance(text, str) for text in cached["texts"])
            ):
                print(f"Читаем тексты из локального кеша: {CACHE_FILE}")
                return cached["texts"]
            print("Настройки источников изменились или кеш устарел, обновляем...")

    texts = []
    for dataset_name, fields, limit in SOURCES:
        texts.extend(load_texts(dataset_name, fields, limit))

    cached = {"version": CACHE_VERSION, "sources": sources, "texts": texts}
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Сначала записываем временный файл, чтобы не оставить неполный кеш.
    temporary_file = CACHE_FILE.with_suffix(".json.tmp")
    temporary_file.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
    temporary_file.replace(CACHE_FILE)
    print(f"Кеш сохранён: {CACHE_FILE}")
    return texts


def load_texts(
    dataset_name: str, fields: tuple[str, ...], limit: int | None
) -> list[str]:
    print(f"Загрузка {dataset_name}...")
    dataset = load_dataset(dataset_name, split="train", streaming=False)
    if limit is not None:
        dataset = dataset.take(limit)

    texts = []
    for row in dataset:
        if fields == ("parts",):
            # Главы Ficbook храним отдельными текстами.
            parts = row["parts"]
            if isinstance(parts, dict):
                candidates = parts["clean_text"]
            else:
                candidates = [part["clean_text"] for part in parts]
        else:
            pieces = []
            for field in fields:
                value = row[field]
                if isinstance(value, str) and value.strip():
                    pieces.append(value.strip())
            candidates = [" ".join(pieces)]

        for text in candidates:
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())

    if not texts:
        raise ValueError(f"Источник {dataset_name} не содержит подходящих текстов")
    print(f"{dataset_name}: загружено {len(texts)} текстов")
    return texts


def main() -> None:
    texts = load_corpus()

    tokenizer = RegexTokenizer(min_frequency=5)
    generator = Markovka(tokenizer, max_length=500, n_gramm=8)
    generator.fit(texts=texts)
    print(f"Всего текстов: {len(texts)}")
    print(f"Размер словаря: {tokenizer.vocab_size}")

    result = generator.generate(prefix="Какую мебель я купила себе домой?")
    print(f"Результат: {result}")


if __name__ == "__main__":
    main()
