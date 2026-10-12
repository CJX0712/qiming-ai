"""启明 · 向量索引层。

**零重写 / 集大成**：
- 主实现：`faiss <https://github.com/facebookresearch/faiss>`_（Meta 出品，
  向量检索世界标准，MIT 许可）。默认 ``IndexFlatIP`` 精确内积索引
  （向量已 L2 归一化，内积 == 余弦），万级以下语料完全精确；
  ``hnsw=True`` 时切换 ``IndexHNSWFlat`` 近似索引，十万级以上推荐。
- 参照实现：numpy 暴力精确检索（矩阵乘 + argpartition）。
  两个独立实现互为交叉验证（参照实现 vs 主实现），是测试套件的
  硬不变量之一。

可验证不变量（见 docs/VERIFICATION.md）：
- I-V1  自检索：库中某向量的 top-1 必为自身，得分 1.0
- I-V2  得分序列单调不增
- I-V3  [faiss Flat] 与 numpy 参照实现 top-k 结果完全一致
- I-V4  k > 库大小时返回全部，不报错
"""

from __future__ import annotations

import logging
from typing import Protocol, Sequence

import numpy as np

logger = logging.getLogger("qiming.vectorstore")


class VectorStore(Protocol):
    """向量存储协议。"""

    dim: int
    backend: str

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None: ...

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]: ...

    def __len__(self) -> int: ...


class NumpyStore:
    """numpy 暴力精确检索（参照实现）。

    ``scores = V @ q`` 全量打分后 argpartition 取 top-k。
    时间复杂度 O(n·d)，空间 O(n·d)——精确、可复现、零依赖。
    """

    backend = "numpy"

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._ids: list[str] = []
        self._vecs = np.zeros((0, dim), dtype=np.float32)

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"向量形状应为 (n, {self.dim})，收到 {vectors.shape}")
        self._ids.extend(ids)
        self._vecs = np.vstack([self._vecs, vectors]) if len(self._ids) > vectors.shape[0] else vectors

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if len(self._ids) == 0:
            return []
        scores = self._vecs @ q
        k = min(k, len(self._ids))
        if k < len(self._ids):
            idx = np.argpartition(scores, -k)[-k:]
        else:
            idx = np.arange(len(self._ids))
        idx = idx[np.argsort(-scores[idx], kind="stable")]
        return [(self._ids[i], float(scores[i])) for i in idx]


class FaissStore:
    """faiss 向量索引（主实现）。"""

    backend = "faiss"

    def __init__(self, dim: int, hnsw: bool = False) -> None:
        import faiss  # 延迟导入

        self.dim = dim
        if hnsw:
            self._index = faiss.IndexHNSWFlat(dim, 32)  # M=32，faiss 推荐默认
        else:
            self._index = faiss.IndexFlatIP(dim)
        self._ids: list[str] = []
        self._backend_name = "faiss(IndexHNSWFlat)" if hnsw else "faiss(IndexFlatIP)"

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"向量形状应为 (n, {self.dim})，收到 {vectors.shape}")
        self._index.add(vectors)
        self._ids.extend(ids)

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        if len(self._ids) == 0:
            return []
        q = np.ascontiguousarray(np.asarray(query, dtype=np.float32).reshape(1, -1))
        k = min(k, len(self._ids))
        scores, idx = self._index.search(q, k)
        out: list[tuple[str, float]] = []
        for s, i in zip(scores[0], idx[0]):
            if i < 0:  # faiss 对不足 k 个结果返回 -1
                continue
            out.append((self._ids[int(i)], float(s)))
        return out


def build_store(backend: str = "auto", dim: int = 0, hnsw: bool = False) -> VectorStore:
    """按配置构造向量库；auto 模式下 faiss 不可用则回退 numpy。"""
    if backend in ("auto", "faiss") and dim > 0:
        try:
            return FaissStore(dim, hnsw=hnsw)
        except Exception as exc:
            logger.warning("faiss 不可用（%s），回退 numpy 精确索引", exc)
            if backend == "faiss":
                raise
    return NumpyStore(dim)
