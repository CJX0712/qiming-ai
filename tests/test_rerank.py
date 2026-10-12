"""重排层测试（I-R1 ~ I-R3）。"""

from __future__ import annotations

import pytest

from qiming.embeddings import HashingEmbedder
from qiming.rerank import CosineReranker, build_reranker


class TestCosineReranker:
    @pytest.fixture(scope="class")
    def reranker(self):
        return CosineReranker(HashingEmbedder(dim=256))

    def test_r2_preserves_candidate_set(self, reranker):
        texts = ["文本甲", "文本乙", "文本丙"]
        scores = reranker.score("查询", texts)
        assert len(scores) == len(texts)

    def test_r3_deterministic(self, reranker):
        s1 = reranker.score("如何重置密码", ["重置密码的方法", "无关内容"])
        s2 = reranker.score("如何重置密码", ["重置密码的方法", "无关内容"])
        assert s1 == s2

    def test_empty_texts(self, reranker):
        assert reranker.score("查询", []) == []

    def test_r1_lexical_relevance_ordering(self, reranker):
        """词法重合度更高者得分更高（哈希嵌入的确定性属性）。"""
        texts = ["密码 重置 邮箱 验证", "完全 无关 的 内容 xyz"]
        scores = reranker.score("密码 重置", texts)
        assert scores[0] > scores[1]


class TestBuildReranker:
    def test_cosine_fallback(self):
        r = build_reranker("cosine", embedder=HashingEmbedder(dim=128))
        assert isinstance(r, CosineReranker)

    def test_auto_falls_back_or_loads(self):
        r = build_reranker("auto", embedder=HashingEmbedder(dim=128))
        assert r.backend in ("cross-encoder", "cosine")
