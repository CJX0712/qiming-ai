"""BM25 稀疏检索测试（I-S1 ~ I-S3）：rank_bm25 主实现 + 内置参照实现交叉验证。

I-S1 金标准语料特意选用纯英文（tokenize 对英文只按词切分，无中文
字/二元组干扰），使 Okapi 公式的各项可手工计算：

    N = 2, df(password) = df(reset) = 1, avgdl = |d1| = 3
    idf = ln(1 + (N - df + 0.5) / (df + 0.5)) = ln(2)
    项分 = idf · f·(k1+1) / (f + k1·(1 - b + b·|d|/avgdl)) = ln(2)·2.5/2.5 = ln(2)
    score(d1) = 2·ln(2) ≈ 1.3862943611198906
"""

from __future__ import annotations

import math

import pytest

from qiming.sparse import BuiltinBM25Index, RankBM25Index, build_sparse_index

CN_DOCS = [
    ("d1", "密码 重置 方法 与 邮箱 验证"),
    ("d2", "天气 预报 服务 与 温度"),
    ("d3", "密码 策略 与 安全 要求"),
]

EN_DOCS = [
    ("d1", "password reset mail"),
    ("d2", "weather forecast today"),
]


class TestBuiltinBM25:
    @pytest.fixture(scope="class")
    def idx(self):
        i = BuiltinBM25Index()
        i.add([d for d, _ in CN_DOCS], [t for _, t in CN_DOCS])
        return i

    def test_i_s2_full_match_ranks_first(self, idx):
        hits = idx.search("密码 重置", k=3)
        assert hits
        assert hits[0][0] == "d1"

    def test_i_s3_empty_index(self):
        assert BuiltinBM25Index().search("任意 查询", 5) == []

    def test_i_s1_hand_computed_golden(self):
        """金标准：英文微型语料，按 Okapi 公式手工计算得分。"""
        i = BuiltinBM25Index()
        i.add([d for d, _ in EN_DOCS], [t for _, t in EN_DOCS])
        hits = dict(i.search("password reset", k=2))
        expected = 2 * math.log(2)  # 详见模块 docstring 推导
        assert "d1" in hits
        assert abs(hits["d1"] - expected) < 1e-9, f"got {hits['d1']}, want {expected}"
        assert hits.get("d2", 0.0) == 0.0


class TestRankBM25:
    @pytest.fixture(scope="class")
    def idx(self):
        i = RankBM25Index()
        i.add([d for d, _ in CN_DOCS], [t for _, t in CN_DOCS])
        return i

    def test_full_match_first(self, idx):
        hits = idx.search("密码 重置", k=3)
        assert hits[0][0] == "d1"

    def test_agreement_with_builtin(self, idx):
        """交叉验证：主实现与内置参照实现对多个查询给出一致排名。"""
        builtin = BuiltinBM25Index()
        builtin.add([d for d, _ in CN_DOCS], [t for _, t in CN_DOCS])
        for q in ["密码", "天气 预报", "安全", "邮箱 验证"]:
            r_main = [d for d, _ in idx.search(q, 3)]
            r_ref = [d for d, _ in builtin.search(q, 3)]
            assert r_main == r_ref, f"query={q}: main={r_main} ref={r_ref}"

    def test_empty_index(self):
        assert RankBM25Index().search("任意", 5) == []


class TestBuildSparseIndex:
    def test_auto(self):
        i = build_sparse_index()
        assert i.backend in ("rank_bm25", "builtin-bm25")
