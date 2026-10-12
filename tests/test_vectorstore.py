"""向量库不变量测试（I-V1 ~ I-V4）：faiss 主实现 vs numpy 参照实现交叉验证。"""

from __future__ import annotations

import numpy as np
import pytest

from qiming.vectorstore import FaissStore, NumpyStore, build_store


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(42)
    mat = rng.standard_normal((50, 64)).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    ids = [f"c{i}" for i in range(50)]
    return ids, mat


@pytest.fixture(scope="module")
def numpy_store(data):
    ids, mat = data
    s = NumpyStore(64)
    s.add(ids, mat)
    return s


@pytest.fixture(scope="module")
def faiss_store(data):
    ids, mat = data
    try:
        s = FaissStore(64)
    except Exception:
        pytest.skip("faiss 不可用")
    s.add(ids, mat)
    return s


class TestNumpyStore:
    def test_i_v1_self_retrieval_top1(self, numpy_store, data):
        _, mat = data
        hits = numpy_store.search(mat[7], 5)
        assert hits[0][0] == "c7"

    def test_i_v1b_self_score_is_one(self, numpy_store, data):
        _, mat = data
        hits = numpy_store.search(mat[13], 1)
        assert abs(hits[0][1] - 1.0) < 1e-5

    def test_i_v2_scores_monotonic(self, numpy_store, data):
        _, mat = data
        hits = numpy_store.search(mat[0], 10)
        scores = [s for _, s in hits]
        assert all(a >= b - 1e-6 for a, b in zip(scores, scores[1:]))

    def test_i_v4_k_larger_than_store(self, numpy_store, data):
        _, mat = data
        assert len(numpy_store.search(mat[0], 999)) == 50

    def test_empty_store_returns_empty(self):
        s = NumpyStore(8)
        assert s.search(np.ones(8, dtype=np.float32), 5) == []

    def test_wrong_dim_raises(self, numpy_store):
        with pytest.raises(ValueError):
            numpy_store.add(["x"], np.ones((1, 3), dtype=np.float32))


class TestFaissStore:
    def test_i_v1_self_retrieval(self, faiss_store, data):
        _, mat = data
        hits = faiss_store.search(mat[7], 5)
        assert hits[0][0] == "c7"

    def test_i_v3_cross_implementation_agreement(self, faiss_store, numpy_store, data):
        """硬不变量：两个独立实现对同一查询给出完全一致的 top-k。"""
        _, mat = data
        for q_idx in [0, 7, 23, 49]:
            r_np = numpy_store.search(mat[q_idx], 10)
            r_fa = faiss_store.search(mat[q_idx], 10)
            assert [i for i, _ in r_np] == [i for i, _ in r_fa]
            assert all(abs(a[1] - b[1]) < 1e-4 for a, b in zip(r_np, r_fa))

    def test_backend_name(self, faiss_store):
        assert "faiss" in faiss_store.backend_name


class TestBuildStore:
    def test_auto_prefers_faiss(self):
        s = build_store("auto", dim=32)
        assert s.dim == 32
        assert s.backend in ("faiss", "numpy")

    def test_forced_numpy(self):
        s = build_store("numpy", dim=32)
        assert isinstance(s, NumpyStore)
