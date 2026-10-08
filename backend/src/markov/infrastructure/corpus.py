import logging

from datasets import load_dataset

# Лимит задаёт количество записей, а не токенов. None — весь источник.
SOURCES = [
    ("Mikimi/russian-wikipedia-top100k", ("summary",), None),
]

logger = logging.getLogger("markov.corpus")


def load_corpus() -> list[str]:
    texts = []
    for dataset_name, fields, limit in SOURCES:
        texts.extend(load_texts(dataset_name, fields, limit))

    return texts


def load_texts(
    dataset_name: str, fields: tuple[str, ...], limit: int | None
) -> list[str]:
    logger.info("Загрузка %s...", dataset_name)
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
    logger.info("%s: загружено %s текстов", dataset_name, len(texts))
    return texts
