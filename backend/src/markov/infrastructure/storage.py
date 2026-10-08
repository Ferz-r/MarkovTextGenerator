"""Versioned model snapshots containing primitive data, never Python objects."""

import gzip
import logging
import os
import pickle
from pathlib import Path
from tempfile import NamedTemporaryFile

logger = logging.getLogger("markov.storage")
MODEL_FILE = Path(__file__).resolve().parents[3] / "data" / "model.pkl.gz"
SNAPSHOT_VERSION = 1


class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        raise pickle.UnpicklingError("Model snapshots cannot contain Python objects")


def read_snapshot(path: Path) -> dict:
    with gzip.open(path, "rb") as source:
        snapshot = PrimitiveUnpickler(source).load()
    if not isinstance(snapshot, dict) or snapshot.get("version") != SNAPSHOT_VERSION:
        raise ValueError("Unsupported model snapshot version")
    return snapshot


def write_snapshot(path: Path, snapshot: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    ) as target:
        temporary = Path(target.name)
        try:
            with gzip.GzipFile(
                fileobj=target, mode="wb", compresslevel=1
            ) as compressed:
                pickle.dump(
                    {"version": SNAPSHOT_VERSION, **snapshot}, compressed, protocol=5
                )
            target.flush()
            os.fsync(target.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    logger.info("Обученная модель сохранена: %s", path)
