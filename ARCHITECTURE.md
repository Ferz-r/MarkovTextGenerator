# Architecture

## Object responsibilities

| Component | Responsibility |
| --- | --- |
| `Tokenizer` | Train the vocabulary and convert text to IDs and back |
| `CharacterTokenizer` / `RegexTokenizer` | Split text into characters or words, numbers, spaces, and punctuation |
| `TransitionIndex` | Store transitions and short-context frequencies compactly |
| `MarkovChain` | Own the corpus, training, generation, topic indexes, and model state |
| `ModelSettings` / `ModelStats` | Represent settings and statistics query results |
| `MarkovService` | Synchronize operations, prepare and publish new state |
| `storage` | Read snapshots and write them atomically to disk |
| `corpus` / `file_corpus` | Load training texts from external sources and files |
| `api` | Validate HTTP requests, return responses, and stream progress events |
| `app.py` | Create the application, load the model, and mount the interface |

## Dependencies

`api` calls `application`; `application` uses `domain` and file storage from
`infrastructure`. `infrastructure` can use shared progress events.
`domain` does not import the API, service, file storage, or web framework.
Settings types use Pydantic but are defined independently of transport schemas.

File loaders and HTTP routes remain functions: they have no state that would
justify a separate class hierarchy. Tokenizers share a single ABC contract;
there is no redundant unused Protocol.

## Model updates

1. The service acquires the lock.
2. It creates a new model and trains it on the new corpus or the combined old and new corpora.
3. When the tokenizer type, context, or frequency changes, it prepares known topics.
4. It saves the candidate to a temporary file and atomically replaces the snapshot.
5. It publishes the new object and settings in memory.

A training or write failure preserves the active model and previous file.
Changing only the generation length saves the new settings before modifying the
active object and requires no training.

Adding texts requires full training to recalculate vocabulary frequencies.
The service does not duplicate the corpus: `MarkovChain` is its sole owner.
The `texts` property returns an immutable snapshot of the sequence.

## Snapshots

The version 1 format is preserved: vocabulary, texts, packed transitions, topics,
and settings. The model restores its state through `from_state`; the index
validates all packed token codes. The service neither reads nor modifies private
model fields. File storage prohibits deserializing Python classes.

## Extension points

A new tokenizer inherits from `Tokenizer`, implements `_split`, and is added to the
model's tokenizer selection. A new corpus source provides a `Callable[[], list[str]]`
for `create_app`. Change persistence and synchronization policies in the service,
HTTP representations in `api`, and the generation algorithm in the model.
