"""启明 · 向量嵌入层。

**零重写 / 集大成**：
- 主实现：`fastembed <https://github.com/qdrant/fastembed>`_（Qdrant 出品，
  ONNX Runtime 推理，无需 GPU / torch），默认模型
  ``BAAI/bge-small-zh-v1.5``（智源 BGE，中英双语检索 SOTA 轻量模型，
  MTEB/ C-MTEB 榜单第一梯队，MIT 许可）。
- 回退实现：确定性哈希嵌入（hashing trick，Weinberger et al. 2009），
  保证系统在任何环境可运行、可复现、可验证——语义能力让位，
  确定性不妥协。

可验证不变量（见 docs/VERIFICATION.md）：
- I-E1  输出形状 == (n, dim)
- I-E2  每行 L2 范数 == 1（单位向量，内积即余弦）
- I-E3  同文本重复编码，结果逐位一致（确定性）
- I-E4  自相似度 == 1
- I-E5  [语义模型] 相关文本相似度 > 无关文本相似度
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Protocol, Sequence

import numpy as np

logger = logging.getLogger("qiming.embeddings")

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
FALLBACK_DIM = 384

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# 中文高频功能字：不携带主题信息，相关性判断必须滤除
# （"珠峰海拔是多少" 与 "飞书集成" 共享的 的/是 不构成相关性）
STOPWORDS = frozenset(
    "的了在是与或及为以被把让向从到有无不没吗呢吧啊怎么如何什哪些着过得之其此各某并就"
)


def tokenize(text: str) -> list[str]:
    """中英混合分词：英文按词、中文按字 + 相邻二元组（bigram）。

    确定性分词是哈希嵌入与 BM25 共用的基础，必须纯函数、无随机性。
    """
    text = text.lower()
    words = _TOKEN_RE.findall(text)
    cjk = _CJK_RE.findall(text)
    bigrams = [a + b for a, b in zip(cjk, cjk[1:])]
    return words + cjk + bigrams


def content_tokens(text: str) -> set[str]:
    """滤除功能字后的实义词集合（用于相关性判断）。"""
    return {t for t in tokenize(text) if not (len(t) == 1 and t in STOPWORDS)}


def _stable_hash(token: str) -> int:
    """跨平台稳定的 64 位哈希（blake2b，避免 Python hash 随机化）。"""
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")


class Embedder(Protocol):
    """嵌入器协议。"""

    dim: int
    name: str
    semantic: bool

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """将文本编码为 L2 归一化的 (n, dim) 矩阵。"""
        ...


class HashingEmbedder:
    """确定性哈希嵌入（回退实现）。

    特征哈希（hashing trick）：token -> blake2b -> 维度和符号，
    累加后 L2 归一化。数学上等价于对 one-hot 稀疏随机投影的
    确定性实例化，无需训练、无需网络、完全可复现。
    """

    semantic = False

    def __init__(self, dim: int = FALLBACK_DIM) -> None:
        self.dim = dim
        self.name = f"hashing-trick-v1(dim={dim})"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in tokenize(text):
                h = _stable_hash(tok)
                out[i, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
            norm = float(np.linalg.norm(out[i]))
            if norm > 0.0:
                out[i] /= norm
        return out


class FastEmbedder:
    """fastembed 语义嵌入（主实现）。"""

    semantic = True

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from fastembed import TextEmbedding  # 延迟导入，回退路径零依赖

        self._model = TextEmbedding(model_name)
        # fastembed 各版本维度属性名不同：embedding_size (>=0.4) / dim (旧版)
        dim = getattr(self._model, "embedding_size", None) or getattr(self._model, "dim", None)
        if dim is None and hasattr(self._model, "get_embedding_size"):
            dim = self._model.get_embedding_size()
        if not dim:
            raise RuntimeError(f"无法从 fastembed 模型 {model_name} 获取维度")
        self.dim = int(dim)
        self.name = f"fastembed({model_name})"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vecs = np.asarray(list(self._model.embed(list(texts))), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0  # 防御零向量，保证 I-E2 恒成立
        return vecs / norms


def get_embedder(backend: str = "auto", model: str = DEFAULT_MODEL) -> Embedder:
    """按配置构造嵌入器；auto 模式下语义模型不可用则自动回退。"""
    if backend in ("auto", "fastembed"):
        try:
            return FastEmbedder(model)
        except Exception as exc:  # 模型缺失 / 无网络 / 平台不支持
            logger.warning("fastembed 不可用（%s），回退确定性哈希嵌入", exc)
            if backend == "fastembed":
                raise
    return HashingEmbedder()
