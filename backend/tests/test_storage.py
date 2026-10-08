import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import create_app
from markov.markov_chain import Markovka
from markov.service import MarkovService
from markov.storage import read_snapshot, write_snapshot
from markov.tokenizer import CharacterTokenizer, RegexTokenizer


class StorageTests(unittest.TestCase):
    def test_restart_restores_changes_without_loading_or_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pkl.gz"
            with TestClient(
                create_app(lambda: ["кот спит."] * 5, model_path=path)
            ) as client:
                client.post("/api/v1/train", json={"texts": ["пёс бежит."] * 5})
                client.post("/api/v1/generate", json={"topic": "кот"})
                settings = client.get("/api/v1/model").json()["settings"]
                settings.update(
                    tokenizer="regex", min_frequency=1, n_gramm=4, max_length=1
                )
                client.post("/api/v1/settings", json=settings)
                stats = client.get("/api/v1/model").json()
            loader = unittest.mock.Mock(
                side_effect=AssertionError("unexpected corpus download")
            )
            app = create_app(loader, model_path=path)
            with (
                patch.object(
                    Markovka, "fit", side_effect=AssertionError("unexpected training")
                ),
                patch.object(
                    Markovka,
                    "_count_transitions",
                    side_effect=AssertionError("unexpected indexing"),
                ),
                TestClient(app) as client,
            ):
                self.assertEqual(client.get("/api/v1/model").json(), stats)
                self.assertIsInstance(app.state.service.tokenizer, RegexTokenizer)
                self.assertEqual(
                    client.post("/api/v1/generate", json={"topic": "кот"}).json()[
                        "text"
                    ],
                    "кот",
                )
            loader.assert_not_called()
            service = MarkovService.load(path)
            service.train(["мышь спит."] * 2)
            restored = MarkovService.load(path)
            self.assertEqual(restored.stats().texts_count, 12)
            self.assertEqual(restored.model.transitions, service.model.transitions)

    def test_character_snapshot_and_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pkl.gz"
            service = MarkovService(model_path=path, min_frequency=1, n_gramm=50)
            service.train(["арбуз", "банан"], replace=True)
            restored = MarkovService.load(path)
            self.assertIsInstance(restored.tokenizer, CharacterTokenizer)
            self.assertEqual(restored.model.transitions, service.model.transitions)
            restored.train(["вишня"], replace=True)
            self.assertEqual(MarkovService.load(path)._texts, ["вишня"])

    def test_corruption_retrains_and_atomic_failure_keeps_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pkl.gz"
            path.write_bytes(b"broken")
            with TestClient(create_app(lambda: ["кот"], model_path=path)) as client:
                self.assertEqual(client.get("/api/v1/model").json()["texts_count"], 1)
            old = path.read_bytes()
            with (
                patch("markov.storage.pickle.dump", side_effect=OSError("disk error")),
                self.assertRaises(OSError),
            ):
                write_snapshot(path, {"settings": {}})
            self.assertEqual(path.read_bytes(), old)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])
            self.assertEqual(read_snapshot(path)["texts"], ["кот"])
