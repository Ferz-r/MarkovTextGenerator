from pathlib import Path

from markov.progress import TrainingProgress


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
        progress = TrainingProgress("Reading file", path.stat().st_size)
        last_offset = 0
        with path.open("r", encoding=encoding) as source:
            for line in source:
                if "\x00" in line:
                    raise CorpusFileError(
                        "The file contains binary data. Choose a plain .txt file."
                    )
                text = line.strip()
                if text:
                    texts.append(text)
                offset = min(source.buffer.tell(), max(0, progress.total - 1))
                if offset > last_offset:
                    progress.advance(offset)
                    last_offset = offset
        if not texts:
            raise CorpusFileError("The file is empty. Add text for training.")
        progress.advance(progress.total)
        return texts

    try:
        return read(encoding)
    except UnicodeDecodeError:
        if encoding != "utf-8-sig":
            raise CorpusFileError(
                "Could not read the file. Save it as UTF-8."
            ) from None
        return read("cp1251")
