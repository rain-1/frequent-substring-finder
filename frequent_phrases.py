"""Frequent repeated phrase finder and CLI tool."""
from __future__ import annotations

import argparse
import json
import re
import string
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple

# Defaults
_DEFAULT_MAX_N = 20
_MAX_UNIQUE_NGRAMS = 50_000_000


@dataclass
class PhraseOccurrence:
    phrase: str
    count: int
    token_spans: List[Tuple[int, int]]
    sentence_indices: List[int]


def split_sentences(text: str) -> List[str]:
    """Naively split text into sentences.

    Sentences are split on '.', '!' or '?' followed by whitespace or end of string.
    Returns trimmed, non-empty sentences.
    """

    pattern = re.compile(r"[.!?](?:\s+|$)")
    parts = pattern.split(text)
    return [part.strip() for part in parts if part.strip()]


def tokenize_sentence(
    sentence: str,
    normalize_case: bool = True,
    strip_punctuation: bool = True,
) -> List[str]:
    """Tokenize a sentence into words.

    Leading/trailing punctuation may be stripped while preserving internal punctuation.
    """

    if normalize_case:
        sentence = sentence.lower()
    tokens: List[str] = []
    for raw_token in sentence.split():
        token = raw_token
        if strip_punctuation:
            token = token.strip(string.punctuation)
        if token:
            tokens.append(token)
    return tokens


def tokenize_document(
    text: str,
    normalize_case: bool = True,
    strip_punctuation: bool = True,
) -> tuple[List[str], List[int]]:
    """Tokenize the document into a flat token list with sentence indices."""

    sentences = split_sentences(text)
    tokens: List[str] = []
    sentence_indices: List[int] = []
    for idx, sentence in enumerate(sentences):
        sentence_tokens = tokenize_sentence(
            sentence, normalize_case=normalize_case, strip_punctuation=strip_punctuation
        )
        tokens.extend(sentence_tokens)
        sentence_indices.extend([idx] * len(sentence_tokens))
    return tokens, sentence_indices


def _validate_parameters(min_words: int, max_words: int | None, min_count: int) -> int:
    if min_words < 1:
        raise ValueError("min_words must be at least 1")
    if min_count < 2:
        raise ValueError("min_count must be at least 2")
    if max_words is not None and max_words < min_words:
        raise ValueError("max_words must be >= min_words when provided")
    if max_words is None:
        return _DEFAULT_MAX_N
    return max_words


def _generate_ngram_positions(
    tokens: List[str], min_words: int, max_n: int, min_count: int
) -> dict[tuple[str, ...], List[int]]:
    ngram_positions: dict[tuple[str, ...], List[int]] = defaultdict(list)
    total_tokens = len(tokens)

    for start in range(total_tokens):
        max_len = min(max_n, total_tokens - start)
        if max_len < min_words:
            continue
        for length in range(min_words, max_len + 1):
            ngram = tuple(tokens[start : start + length])
            ngram_positions[ngram].append(start)
        if len(ngram_positions) > _MAX_UNIQUE_NGRAMS:
            raise MemoryError(
                "N-gram space too large; consider lowering max_words or using smaller input"
            )

    return {ng: pos for ng, pos in ngram_positions.items() if len(pos) >= min_count}


def _filter_maximal_ngrams(
    ngram_positions: dict[tuple[str, ...], List[int]], min_words: int
) -> List[tuple[tuple[str, ...], List[int]]]:
    # Group by count
    grouped: defaultdict[int, List[tuple[tuple[str, ...], List[int]]]] = defaultdict(list)
    for ngram, positions in ngram_positions.items():
        grouped[len(positions)].append((ngram, positions))

    maximal: List[tuple[tuple[str, ...], List[int]]] = []

    for count, items in grouped.items():
        sorted_items = sorted(items, key=lambda x: len(x[0]), reverse=True)
        covered_spans: set[tuple[int, int]] = set()

        for ngram, positions in sorted_items:
            length = len(ngram)
            spans = [(start, start + length - 1) for start in positions]
            if all(span in covered_spans for span in spans):
                continue
            maximal.append((ngram, positions))
            for start in positions:
                for offset in range(length):
                    remaining = length - offset
                    for sub_len in range(min_words, remaining + 1):
                        span = (start + offset, start + offset + sub_len - 1)
                        covered_spans.add(span)

    return maximal


def _build_phrase_occurrences(
    tokens: List[str],
    sentence_indices: List[int],
    maximal_ngrams: List[tuple[tuple[str, ...], List[int]]],
) -> List[PhraseOccurrence]:
    occurrences: List[PhraseOccurrence] = []
    for ngram, positions in maximal_ngrams:
        phrase_text = " ".join(ngram)
        spans = [(start, start + len(ngram) - 1) for start in positions]
        sentences = sorted({sentence_indices[start] for start in positions})
        occurrences.append(
            PhraseOccurrence(
                phrase=phrase_text,
                count=len(positions),
                token_spans=spans,
                sentence_indices=sentences,
            )
        )
    return occurrences


def find_repeated_phrases(
    text: str,
    min_words: int = 4,
    max_words: int | None = 10,
    min_count: int = 2,
    normalize_case: bool = True,
    strip_punctuation: bool = True,
) -> List[PhraseOccurrence]:
    """Find repeated contiguous word sequences in text."""

    if not text or text.isspace():
        return []

    max_n = _validate_parameters(min_words, max_words, min_count)

    tokens, sentence_indices = tokenize_document(
        text, normalize_case=normalize_case, strip_punctuation=strip_punctuation
    )

    if len(tokens) < min_words:
        return []

    ngram_positions = _generate_ngram_positions(tokens, min_words, max_n, min_count)
    maximal = _filter_maximal_ngrams(ngram_positions, min_words)
    occurrences = _build_phrase_occurrences(tokens, sentence_indices, maximal)

    return sorted(
        occurrences,
        key=lambda occ: (-occ.count, -len(occ.phrase.split()), occ.phrase),
    )


def _format_text_output(occurrences: List[PhraseOccurrence]) -> str:
    blocks = []
    for occ in occurrences:
        block = [
            f'Phrase: "{occ.phrase}"',
            f"Count: {occ.count}",
            f"Length: {len(occ.phrase.split())} words",
            "Sentence indices: " + ", ".join(str(idx) for idx in occ.sentence_indices),
            "---",
        ]
        blocks.append("\n".join(block))
    return "\n".join(blocks)


def _parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find repeated phrases in text.")
    parser.add_argument("--input", required=True, help="Path to input text file or '-' for stdin")
    parser.add_argument("--min-words", type=int, default=4, dest="min_words")
    parser.add_argument("--max-words", type=int, dest="max_words")
    parser.add_argument("--min-count", type=int, default=2, dest="min_count")
    parser.add_argument("--no-normalize-case", action="store_false", dest="normalize_case")
    parser.add_argument("--no-strip-punctuation", action="store_false", dest="strip_punctuation")
    parser.add_argument("--top-k", type=int, dest="top_k")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args(args)


def _load_text(path: str) -> str:
    if path == "-":
        import sys

        return sys.stdin.read()
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    text = _load_text(args.input)

    occurrences = find_repeated_phrases(
        text,
        min_words=args.min_words,
        max_words=args.max_words,
        min_count=args.min_count,
        normalize_case=args.normalize_case,
        strip_punctuation=args.strip_punctuation,
    )

    if args.top_k is not None:
        occurrences = occurrences[: args.top_k]

    if args.as_json:
        serializable = [
            {
                "phrase": occ.phrase,
                "count": occ.count,
                "token_spans": occ.token_spans,
                "sentence_indices": occ.sentence_indices,
            }
            for occ in occurrences
        ]
        print(json.dumps(serializable, indent=2))
    else:
        print(_format_text_output(occurrences))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
