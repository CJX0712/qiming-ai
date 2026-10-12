"""启明 · BM25 稀疏检索层。

**零重写 / 集大成**：
- 主实现：`rank_bm25 <https://github.com/dorianbrown/rank_bm25>`_
  （BM25Okapi，世界标准纯 Python 实现，MIT 许可）。
- 回退实现：内置 BM25Okapi（Robertson & Zaragoza, 2000 公式直译），
  与主实现互为交叉验证。

BM25 公式（Okapi 变体，k1=1.5, b=0.75）::

    idf(t)  = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
    score   = Σ_t idf(t) · f(t,d)·(k1+1) / (f(t,d) + k1·(1 - b + b·|d|/avgdl))

可验证不变量（见 docs/VERIFICATION.md）：
- I-S1  手算金标准：对微型语料，实现得分与手工计算结果一致（±1e-9）
- I-S2  查询词全部命中某文档时，该文档排名第一
- I-S3  空库检索返回空列表，不抛异常
"""

from __future__ import annotations

import math
from typing import Protocol, Sequence

from qiming.embeddings import tokenize


class SparseIndex(Protocol):
    backend: str

    def add(self, ids: Sequence[str], texts: Sequence[str]) -> None: ...

    def search(self, query: str, k: int) -> list[tuple[str, float]]: ...

    def __len__(self) -> int: ...


class RankBM25Index:
    """rank_bm25 主实现。"""

    backend = "rank_bm25"

    def __init__(self) -> None:
        from rank_bm25 import BM25Okapi

        self._bm25_cls = BM25Okapi
        self._index = None
        self._ids: list[str] = []

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, ids: Sequence[str], texts: Sequence[str]) -> None:
        corpus = [tokenize(t) for t in texts]
        self._index = self._bm25_cls(corpus)
        self._ids = list(ids)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self._index is None or not self._ids:
            return []
        scores = self._index.get_scores(tokenize(query))
        k = min(k, len(self._ids))
        idx = sorted(range(len(self._ids)), key=lambda i: -scores[i])[:k]
        return [(self._ids[i], float(scores[i])) for i in idx if scores[i] > 0.0]


class BuiltinBM25Index:
    """内置 BM25Okapi 回退实现（公式直译，与 rank_bm25 互为参照）。"""

    backend = "builtin-bm25"

    K1 = 1.5
    B = 0.75

    def __init__(self) -> None:
        self._docs: list[list[str]] = []
        self._ids: list[str] = []
        self._df: dict[str, int] = {}
        self._avgdl = 0.0

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, ids: Sequence[str], texts: Sequence[str]) -> None:
        self._ids = list(ids)
        self._docs = [tokenize(t) for t in texts]
        self._df = {}
        for doc in self._docs:
            for term in set(doc):
                self._df[term] = self._df.get(term, 0) + 1
        self._avgdl = sum(len(d) for d in self._docs) / max(len(self._docs), 1)

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        n = len(self._docs)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if not self._ids:
            return []
        q_terms = tokenize(query)
        results: list[tuple[str, float]] = []
        for doc_id, doc in zip(self._ids, self._docs):
            dl = len(doc)
            score = 0.0
            for t in set(q_terms):
                f = doc.count(t)
                if f == 0:
                    continue
                idf = self._idf(t)
                score += idf * (f * (self.K1 + 1.0)) / (f + self.K1 * (1.0 - self.B + self.B * dl / max(self._avgdl, 1e-9)))
            if score > 0.0:
                results.append((doc_id, score))
        results.sort(key=lambda x: -x[1])
        return results[: min(k, len(results))]


def build_sparse_index(backend: str = "auto") -> SparseIndex:
    """按配置构造稀疏索引；auto 模式下 rank_bm25 不可用则回退内置实现。"""
    if backend in ("auto", "rank_bm25"):
        try:
            return RankBM25Index()
        except Exception:
            if backend == "rank_bm25":
                raise
    return BuiltinBM25Index()
