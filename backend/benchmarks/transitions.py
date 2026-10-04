"""Run each implementation in a fresh process: python benchmarks/transitions.py."""

import json
import logging
import resource
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

from markov.markov_chain import Markovka
from markov.tokenizer import CharacterTokenizer

logging.disable(logging.CRITICAL)


def measure(method: str) -> dict:
    corpus = json.loads(
        (Path(__file__).resolve().parents[1] / "data/texts.json").read_text()
    )["texts"]
    # Bound the legacy implementation's allocation to keep the benchmark small.
    texts = []
    remaining = 20000
    for text in corpus:
        if remaining <= 0:
            break
        part = text[:remaining]
        if part:
            texts.append(part)
            remaining -= len(part)
    del corpus
    tokenizer = CharacterTokenizer(min_frequency=3)
    tokenizer.train(texts)
    model = Markovka(tokenizer, n_gramm=50)
    started = perf_counter()
    if method == "legacy":
        transitions = defaultdict(lambda: defaultdict(int))
        for text in texts:
            tokens = [tokenizer.bos_id] * 49 + tokenizer.encode(text)
            for i in range(50, len(tokens)):
                for size in range(1, 51):
                    transitions[tuple(tokens[i - size : i])][tokens[i]] += 1
    else:
        transitions = model._count_transitions(texts)
    elapsed = perf_counter() - started
    return {
        "method": method,
        "characters": sum(map(len, texts)),
        "texts": len(texts),
        "ngramm": 50,
        "seconds": round(elapsed, 3),
        "contexts": len(transitions),
        "peak_rss_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024 * 1024 if sys.platform == "darwin" else 1024),
            1,
        ),
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.stdout.write(json.dumps(measure(sys.argv[1])) + "\n")
    else:
        for method in ("legacy", "indexed"):
            subprocess.run([sys.executable, __file__, method], check=True)
