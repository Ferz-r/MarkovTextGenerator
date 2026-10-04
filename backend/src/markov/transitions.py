from array import array
from bisect import bisect_left
from collections import OrderedDict, defaultdict


class TransitionIndex:
    """Exact backoff counts indexed by a packed, reversed full context.

    A record contains one full context and its following token. All records
    sharing a shorter reversed prefix form a contiguous range in sorted order.
    This avoids storing a separate tuple/dictionary for every context length.
    """

    def __init__(
        self,
        records: dict[int, int],
        token_ids: list[int],
        context_size: int,
    ):
        self._token_ids = token_ids
        self._codes = {token: code for code, token in enumerate(token_ids)}
        self._bits = max(1, (len(token_ids) - 1).bit_length())
        self._mask = (1 << self._bits) - 1
        self._context_size = context_size
        self._keys = sorted(records)
        self._counts = array("Q", (records[key] for key in self._keys))
        self._cache: OrderedDict[tuple[int, int], dict[int, int]] = OrderedDict()
        self._contexts_count = self._count_contexts()

    def _count_contexts(self) -> int:
        # Each new full context adds prefixes beyond the common prefix with
        # its predecessor. Fixed-width token codes make this exact.
        total = 0
        previous = None
        width = self._bits * self._context_size
        for record in self._keys:
            context = record >> self._bits
            if context == previous:
                continue
            common = (
                0
                if previous is None
                else (width - (context ^ previous).bit_length()) // self._bits
            )
            total += self._context_size - common
            previous = context
        return total

    def get(self, context: tuple[int, ...], default=None):
        size = len(context)
        if not 1 <= size <= self._context_size:
            return default
        prefix = 0
        for token in reversed(context):
            code = self._codes.get(token)
            if code is None:
                return default
            prefix = (prefix << self._bits) | code
        key = (size, prefix)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        shift = self._bits * (self._context_size + 1 - size)
        start = bisect_left(self._keys, prefix << shift)
        end = bisect_left(self._keys, (prefix + 1) << shift)
        if start == end:
            return default
        frequencies: dict[int, int] = {}
        for index in range(start, end):
            token = self._token_ids[self._keys[index] & self._mask]
            frequencies[token] = frequencies.get(token, 0) + self._counts[index]
        self._cache[key] = frequencies
        if len(self._cache) > 512:
            self._cache.popitem(last=False)
        return frequencies

    def copy(self) -> dict[tuple[int, ...], dict[int, int]]:
        """Materialize the legacy table only for explicit inspection/export."""
        result = defaultdict(lambda: defaultdict(int))
        for record, count in zip(self._keys, self._counts, strict=True):
            token = self._token_ids[record & self._mask]
            packed = record >> self._bits
            reversed_context = []
            for size in range(1, self._context_size + 1):
                code = (
                    packed >> (self._bits * (self._context_size - size))
                ) & self._mask
                reversed_context.append(self._token_ids[code])
                result[tuple(reversed(reversed_context))][token] += count
        return {context: dict(counts) for context, counts in result.items()}

    def __len__(self) -> int:
        return self._contexts_count
