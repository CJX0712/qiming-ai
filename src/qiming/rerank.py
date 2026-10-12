"""启明 · Cross-Encoder 重排层。

**零重写 / 集大成**：
- 主实现：fastembed ``TextCrossEncoder``（ONNX 推理，无需 torch），
  默认 ``BAAI/bge-reranker-base``（BGE 系列重排模型，中文重排
  SOTA 第一梯队，MIT 许可）。
- 回退实现：双编码器余弦重排（用嵌入层直接打分），零额外依赖。

检索两阶段（retrieve-then-rerank）是当前 RAG 的标准范式：
混合检索召回候选（高 recall），cross-encoder 精排（高 precision）。

可验证不变量（见 docs/VERIFICATION.md）：
- I-R1  [语义模型] 相关文档得分 > 无关文档得分
- I-R2  重排不改变候选集合，只改变顺序
- I-R3  相同输入重复打分，结果一致（确定性）
"""

from __future__ import annotations

import logging
from typing import Protocol, Sequence

import numpy as np

logger = logging.getLogger("qiming.rerank")

# fastembed >= 0.9 的 TextCrossEncoder 注册表内模型（list_supported_models）
# bge-reranker-base：中文重排第一梯队，ONNX 推理无需 torch
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


class Reranker(Protocol):
    backend: str

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """对 (query, text) 对逐个打分，分数越高越相关。"""
        ...


class CrossEncoderReranker:
    """fastembed cross-encoder 重排（主实现）。"""

    backend = "cross-encoder"

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        try:
            from fastembed import TextCrossEncoder  # fastembed < 0.9 顶层导出
        except ImportError:
            from fastembed.rerank.cross_encoder.text_cross_encoder import TextCrossEncoder  # fastembed >= 0.9

        self._model = TextCrossEncoder(model_name)
        self.name = f"fastembed-cross-encoder({model_name})"

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        pairs = [[query, t] for t in texts]
        return [float(s) for s in self._model.rerank(query, list(texts))]


class CosineReranker:
    """双编码器余弦重排（回退实现）。"""

    backend = "cosine"

    def __init__(self, embedder) -> None:
        self._embedder = embedder
        self.name = f"cosine({embedder.name})"

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        qv = self._embedder.encode([query])[0]
        tv = self._embedder.encode(list(texts))
        return [float(s) for s in tv @ qv]


def build_reranker(backend: str = "auto", model: str = DEFAULT_RERANKER_MODEL, embedder=None) -> Reranker:
    """按配置构造重排器；auto 模式下 cross-encoder 不可用则回退余弦。"""
    if backend in ("auto", "cross-encoder"):
        try:
            return CrossEncoderReranker(model)
        except Exception as exc:
            logger.warning("cross-encoder 不可用（%s），回退余弦重排", exc)
            if backend == "cross-encoder":
                raise
    if embedder is None:
        raise ValueError("cosine 回退重排需要 embedder")
    return CosineReranker(embedder)
