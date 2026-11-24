"""Token counting and token-span extraction.

Uses tiktoken (``cl100k_base``) when its BPE file is available. If tiktoken
cannot load (for example, offline on first run), it falls back to a
deterministic regex tokenizer, so experiments still run. The backend in use is
reported in every corpus and run summary.
"""

from __future__ import annotations

import bisect
import re
from functools import lru_cache

_REGEX = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class Tokenizer:
    name: str = "regex"

    def spans(self, text: str) -> list[tuple[int, int]]:
        return [(m.start(), m.end()) for m in _REGEX.finditer(text)]

    def count(self, text: str) -> int:
        return sum(1 for _ in _REGEX.finditer(text))


class TiktokenTokenizer(Tokenizer):
    def __init__(self, encoding: str = "cl100k_base") -> None:
        import tiktoken

        self._enc = tiktoken.get_encoding(encoding)
        self.name = f"tiktoken:{encoding}"

    def count(self, text: str) -> int:
        return len(self._enc.encode(text, disallowed_special=()))

    def spans(self, text: str) -> list[tuple[int, int]]:
        ids = self._enc.encode(text, disallowed_special=())
        decoded, offsets = self._enc.decode_with_offsets(ids)
        if decoded != text:  # pragma: no cover - only on lossy decode
            return super().spans(text)
        # A token that starts mid-character (CJK, emoji) shares its offset with the previous
        # token. Keep it as a zero-width span: dropping it undercounts tokens and lets chunks
        # silently exceed their size cap.
        ends = offsets[1:] + [len(text)]
        return list(zip(offsets, ends))


@lru_cache(maxsize=1)
def get_tokenizer() -> Tokenizer:
    try:
        return TiktokenTokenizer()
    except Exception:  # network / missing BPE file
        return Tokenizer()


class TokenIndex:
    """Precomputed token spans for one document; answers range counts in O(log n)."""

    def __init__(self, text: str, tokenizer: Tokenizer | None = None) -> None:
        self.text = text
        self.spans = (tokenizer or get_tokenizer()).spans(text)
        self.starts = [s for s, _ in self.spans]

    def count(self, start: int, end: int) -> int:
        return bisect.bisect_left(self.starts, end) - bisect.bisect_left(self.starts, start)

    def __len__(self) -> int:
        return len(self.spans)
