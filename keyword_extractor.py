"""High-performance keyword extraction with jieba.

This module wraps jieba's TF-IDF and TextRank extractors and reuses analyzer
instances to reduce repeated initialization overhead in batch scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import jieba.analyse


@dataclass(frozen=True)
class Keyword:
    word: str
    weight: float


class KeywordExtractor:
    """Extract Chinese keywords with reusable analyzer instances."""

    def __init__(self, allow_pos: Sequence[str] = ("n", "nr", "ns"), top_k: int = 10) -> None:
        self.allow_pos = tuple(allow_pos)
        self.top_k = top_k
        self._tfidf = jieba.analyse.TFIDF()
        self._textrank = jieba.analyse.TextRank()

    def extract_tfidf(self, text: str, top_k: int | None = None) -> list[Keyword]:
        """Extract keywords via TF-IDF."""
        limit = top_k or self.top_k
        rows = self._tfidf.extract_tags(
            text,
            topK=limit,
            withWeight=True,
            allowPOS=self.allow_pos,
        )
        return [Keyword(word=w, weight=weight) for w, weight in rows]

    def extract_textrank(self, text: str, top_k: int | None = None) -> list[Keyword]:
        """Extract keywords via TextRank."""
        limit = top_k or self.top_k
        rows = self._textrank.textrank(
            text,
            topK=limit,
            withWeight=True,
            allowPOS=self.allow_pos,
        )
        return [Keyword(word=w, weight=weight) for w, weight in rows]


def _format_result(title: str, items: Iterable[Keyword]) -> str:
    lines = [title]
    lines.extend(f"- {item.word}: {item.weight:.6f}" for item in items)
    return "\n".join(lines)


def main() -> None:
    content = "该同学来电反映学校食堂趁刮台风涨价，建议拨打12345"
    extractor = KeywordExtractor()

    print(_format_result("TF-IDF:", extractor.extract_tfidf(content)))
    print()
    print(_format_result("TextRank:", extractor.extract_textrank(content)))


if __name__ == "__main__":
    main()
