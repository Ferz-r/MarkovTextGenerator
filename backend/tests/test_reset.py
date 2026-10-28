import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from markov.app import create_app
from markov.application.service import MarkovService


class ResetTests(unittest.TestCase):
    def test_reset_restores_startup_corpus_settings_and_disk_repeatedly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pkl.gz"
            app = create_app(lambda: ["Original cat sleeps."] * 3, model_path=path)
            with TestClient(app) as client:
                service = app.state.service
                stats = service.stats()
                state = service.model.export_state()
                for endpoint in ("/api/v1/reset", "/api/v1/reset/stream"):
                    client.post("/api/v1/generate", json={"topic": "cat"})
                    client.post(
                        "/api/v1/train",
                        json={"texts": ["New dog runs."] * 3, "replace": True},
                    )
                    settings = stats.settings.model_copy(
                        update={"tokenizer": "character", "max_length": 1}
                    )
                    service.configure(settings)
                    response = client.post(endpoint)
                    self.assertEqual(response.status_code, 200)
                    if endpoint.endswith("/stream"):
                        self.assertIn("event: complete", response.text)
                    else:
                        self.assertEqual(response.json(), stats.model_dump())
                    self.assertEqual(service.model.export_state(), state)
                    self.assertEqual(service.stats(), stats)
                    self.assertEqual(
                        MarkovService.load(path).model.export_state(), state
                    )
                    self.assertEqual(
                        client.post(
                            "/api/v1/generate", json={"topic": "dog"}
                        ).status_code,
                        422,
                    )
                    self.assertEqual(
                        client.post(
                            "/api/v1/generate", json={"topic": "cat"}
                        ).status_code,
                        200,
                    )

    def test_loaded_snapshot_is_baseline_without_downloading_or_retraining(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pkl.gz"
            initial = MarkovService(model_path=path, min_frequency=1)
            initial.train(["Saved cat sleeps."], replace=True)
            loader = unittest.mock.Mock(
                side_effect=AssertionError("unexpected download")
            )
            app = create_app(loader, model_path=path)
            with TestClient(app) as client:
                state = app.state.service.model.export_state()
                client.post(
                    "/api/v1/train", json={"texts": ["dog runs."], "replace": True}
                )
                with patch(
                    "markov.domain.model.MarkovChain.fit",
                    side_effect=AssertionError("unexpected training"),
                ):
                    self.assertEqual(client.post("/api/v1/reset").status_code, 200)
                self.assertEqual(app.state.service.model.export_state(), state)
            loader.assert_not_called()

    def test_failed_reset_keeps_current_model_file_and_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pkl.gz"
            with TestClient(
                create_app(lambda: ["cat sleeps."] * 3, model_path=path)
            ) as client:
                service = client.app.state.service
                initial = service.model.export_state()
                service.train(["dog runs."] * 3, replace=True)
                current = service.model
                old_file = path.read_bytes()
                with patch(
                    "markov.infrastructure.storage.pickle.dump",
                    side_effect=OSError("disk full"),
                ):
                    response = client.post("/api/v1/reset/stream")
                self.assertIn("event: error", response.text)
                self.assertIs(service.model, current)
                self.assertEqual(path.read_bytes(), old_file)
                service.reset()
                self.assertEqual(service.model.export_state(), initial)

    def test_reset_in_memory_app_and_initial_snapshot_cleanup(self):
        app = create_app(lambda: ["cat sleeps."] * 3)
        with TestClient(app) as client:
            service = app.state.service
            baseline = service._initial_snapshot_path
            client.post("/api/v1/train", json={"texts": ["dog runs."], "replace": True})
            self.assertEqual(client.post("/api/v1/reset").status_code, 200)
            self.assertEqual(service.model.texts, ("cat sleeps.",) * 3)
            self.assertTrue(baseline.exists())
        self.assertFalse(baseline.exists())
