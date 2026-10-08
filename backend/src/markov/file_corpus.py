from pathlib import Path

from markov.logging import TrainingProgress


class CorpusFileError(ValueError):
    pass


def load_file_texts(path: Path) -> list[str]:
    """Read each nonempty line as a training text on the server without building one giant browser string."""
    with path.open("rb") as source:
        bom = source.read(3)
    encoding = "utf-8-sig"
    if bom.startswith((b"\xff\xfe", b"\xfe\xff")):
        encoding = "utf-16"

    def read(encoding: str) -> list[str]:
        texts = []
        progress = TrainingProgress("Чтение файла", path.stat().st_size)
        last_offset = 0
        with path.open("r", encoding=encoding) as source:
            for line in source:
                if "\x00" in line:
                    raise CorpusFileError(
                        "Файл содержит двоичные данные. Выберите обычный .txt."
                    )
                text = line.strip()
                if text:
                    texts.append(text)
                offset = min(source.buffer.tell(), max(0, progress.total - 1))
                if offset > last_offset:
                    progress.advance(offset)
                    last_offset = offset
        if not texts:
            raise CorpusFileError("Файл пустой. Добавьте текст для обучения.")
        progress.advance(progress.total)
        return texts

    try:
        return read(encoding)
    except UnicodeDecodeError:
        if encoding != "utf-8-sig":
            raise CorpusFileError(
                "Не удалось прочитать файл. Сохраните его в UTF-8."
            ) from None
        return read("cp1251")
