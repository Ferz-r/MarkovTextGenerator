import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from markov.application.service import MarkovService
from markov.domain.model import MarkovChain
from markov.domain.tokenizers.character import CharacterTokenizer
from markov.domain.transitions import TransitionIndex


class ServiceTransactionTests(unittest.TestCase):
    def test_training_failure_preserves_model_vocabulary_corpus_and_topics(self):
        service = MarkovService(min_frequency=1)
        service.train(["кот спит"], replace=True)
        service.generate("", "кот")
        original = service.model
        state = original.export_state()
        with patch.object(MarkovChain, "_count_transitions", side_effect=RuntimeError):
            with self.assertRaises(RuntimeError):
                service.train(["пёс бежит"], replace=True)
        self.assertIs(service.model, original)
        self.assertEqual(service.model.export_state(), state)

    def test_save_failure_preserves_memory_and_disk_for_all_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pkl.gz"
            service = MarkovService(model_path=path, min_frequency=1)
            service.train(["кот спит"], replace=True)
            original = service.model
            state = original.export_state()
            stats = service.stats()
            old_file = path.read_bytes()
            for operation in (
                lambda: service.train(["пёс бежит"], replace=True),
                lambda: service.train(["пёс бежит"]),
                lambda: service.configure(
                    stats.settings.model_copy(update={"n_gramm": 2})
                ),
                lambda: service.configure(
                    stats.settings.model_copy(update={"max_length": 1})
                ),
            ):
                with patch(
                    "markov.infrastructure.storage.pickle.dump", side_effect=OSError
                ):
                    with self.assertRaises(OSError):
                        operation()
                self.assertIs(service.model, original)
                self.assertEqual(service.model.export_state(), state)
                self.assertEqual(service.stats(), stats)
                self.assertEqual(path.read_bytes(), old_file)
                self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_invalid_context_codes_in_snapshot_are_rejected(self):
        # Three tokens use two bits; code 3 is invalid in the context as well as the target.
        with self.assertRaises(ValueError):
            TransitionIndex.from_state(
                {
                    "token_ids": [0, 1, 2],
                    "context_size": 1,
                    "keys": [12],
                    "counts": [1],
                    "contexts_count": 1,
                }
            )

    def test_constructor_rejects_invalid_generation_length(self):
        for length in (0, -1):
            with self.assertRaises(ValueError):
                MarkovChain(CharacterTokenizer(), max_length=length)

    def test_exported_state_does_not_expose_internal_lists(self):
        service = MarkovService(min_frequency=1)
        service.train(["кот"])
        original = service.model.export_state()
        exported = service.model.export_state()
        exported["texts"].clear()
        exported["transitions"]["keys"].clear()
        exported["transitions"]["token_ids"].clear()
        self.assertEqual(service.model.export_state(), original)
