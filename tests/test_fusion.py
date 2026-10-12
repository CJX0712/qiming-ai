"""RRF 融合不变量测试（I-F1 ~ I-F4）。"""

from __future__ import annotations

from qiming.fusion import rrf_fuse


class TestRRF:
    def test_i_f1_self_fusion_preserves_order(self):
        r = ["a", "b", "c", "d"]
        fused = rrf_fuse([r, r])
        assert [d for d, _ in fused] == r

    def test_i_f3_symmetry(self):
        r1, r2 = ["a", "b", "c"], ["c", "b", "a"]
        assert rrf_fuse([r1, r2]) == rrf_fuse([r2, r1])

    def test_i_f4_single_ranking_score(self):
        fused = dict(rrf_fuse([["x", "y", "z"]]))
        assert abs(fused["x"] - 1 / 61) < 1e-12
        assert abs(fused["y"] - 1 / 62) < 1e-12
        assert abs(fused["z"] - 1 / 63) < 1e-12

    def test_i_f2_union_top_consistency(self):
        """任一路的第一名，融合分不低于任一路的第二名。"""
        r1, r2 = ["a", "b"], ["c", "a"]
        fused = rrf_fuse([r1, r2])
        score = dict(fused)
        assert score["a"] >= score["b"]
        assert score["a"] >= score["c"]

    def test_disjoint_rankings_union(self):
        fused = rrf_fuse([["a", "b"], ["c", "d"]])
        assert {d for d, _ in fused} == {"a", "b", "c", "d"}

    def test_three_way_fusion(self):
        fused = rrf_fuse([["a"], ["a"], ["b"]])
        assert fused[0][0] == "a"
        assert abs(fused[0][1] - 2 / 61) < 1e-12

    def test_invalid_k_raises(self):
        import pytest

        with pytest.raises(ValueError):
            rrf_fuse([["a"]], k=0)

    def test_empty_input(self):
        assert rrf_fuse([]) == []
