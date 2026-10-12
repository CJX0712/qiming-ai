"""嵌入层不变量测试（I-E1 ~ I-E5）。

I-E5（语义区分度）在语义模型可用时执行，否则跳过——
回退的哈希嵌入只保证确定性与几何不变量，不保证语义。
"""

from __future__ import annotations

import numpy as np
import pytest

from qiming.embeddings import HashingEmbedder, get_embedder


class TestHashingEmbedder:
    @pytest.fixture(scope="class")
    def emb(self):
        return HashingEmbedder(dim=256)

    def test_i_e1_shape(self, emb):
        vecs = emb.encode(["文本一", "text two", "三"])
        assert vecs.shape == (3, 256)
        assert vecs.dtype == np.float32

    def test_i_e2_unit_norm(self, emb):
        vecs = emb.encode(["任意文本内容", "another text"])
        norms = np.linalg.norm(vecs, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_i_e3_determinism(self, emb):
        a = emb.encode(["同样的输入"])
        b = emb.encode(["同样的输入"])
        assert np.array_equal(a, b)

    def test_i_e4_self_similarity(self, emb):
        vecs = emb.encode(["测试文本", "测试文本"])
        assert abs(float(vecs[0] @ vecs[1]) - 1.0) < 1e-6

    def test_empty_input(self, emb):
        assert emb.encode([]).shape == (0, 256)

    def test_semantic_flag(self, emb):
        assert emb.semantic is False


class TestSemanticEmbedder:
    """语义模型可用时才运行（需要已下载 fastembed 模型）。"""

    @pytest.fixture(scope="class")
    def emb(self):
        try:
            return get_embedder("fastembed")
        except Exception:
            pytest.skip("fastembed 语义模型不可用（未下载或离线）")

    def test_i_e5_semantic_discrimination(self, emb):
        query = "如何重置登录密码"
        related = "用户可以在设置页面重置登录密码，重置链接会发送到注册邮箱。"
        unrelated = "今天天气晴朗，适合外出郊游和爬山。"
        vecs = emb.encode([query, related, unrelated])
        sim_related = float(vecs[0] @ vecs[1])
        sim_unrelated = float(vecs[0] @ vecs[2])
        assert sim_related > sim_unrelated, f"related={sim_related:.3f} unrelated={sim_unrelated:.3f}"

    def test_i_e1_e2_hold_for_semantic(self, emb):
        vecs = emb.encode(["中文句子", "english sentence"])
        assert vecs.shape[1] == emb.dim
        assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
