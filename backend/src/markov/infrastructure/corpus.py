import logging

from datasets import load_dataset

# The limit counts records, not tokens. None uses the entire source.
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
    logger.info("Loading %s...", dataset_name)
    dataset = load_dataset(dataset_name, split="train", streaming=False)
    if limit is not None:
        dataset = dataset.take(limit)

    texts = []
    for row in dataset:
        if fields == ("parts",):
            # Store Ficbook chapters as separate texts.
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
        raise ValueError(f"Source {dataset_name} contains no suitable texts")
    logger.info("%s: loaded %s texts", dataset_name, len(texts))
    return texts
