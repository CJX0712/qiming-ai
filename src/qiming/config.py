"""启明 · 配置层。

所有参数均可通过环境变量覆盖，前缀 ``QIMING_``，例如::

    QIMING_EMBED_MODEL=BAAI/bge-small-en-v1.5
    QIMING_EMBED_BACKEND=hashing          # 强制离线确定性嵌入
    QIMING_INDEX_BACKEND=numpy            # 强制暴力精确索引
    QIMING_LLM_BACKEND=openai
    QIMING_LLM_BASE_URL=https://api.deepseek.com
    QIMING_LLM_API_KEY=sk-...
    QIMING_LLM_MODEL=deepseek-chat

设计原则：默认值让系统在任何机器上开箱即跑（语义嵌入不可用时自动
退化为确定性哈希嵌入，faiss 不可用时退化为 numpy 精确索引），
同时保留接入世界顶级推理服务的完整能力。
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _get(key: str, default: str) -> str:
    return os.environ.get(f"QIMING_{key}", default)


def _get_int(key: str, default: int) -> int:
    return int(_get(key, str(default)))


def _get_float(key: str, default: float) -> float:
    return float(_get(key, str(default)))


@dataclass
class ChunkConfig:
    """文本分块参数。"""

    chunk_size: int = 400
    chunk_overlap: int = 80

    @classmethod
    def from_env(cls) -> "ChunkConfig":
        return cls(
            chunk_size=_get_int("CHUNK_SIZE", 400),
            chunk_overlap=_get_int("CHUNK_OVERLAP", 80),
        )


@dataclass
class EmbedConfig:
    """嵌入层配置。

    backend: ``auto`` | ``fastembed`` | ``hashing``
    model:   fastembed 模型 ID（默认 bge-small-zh-v1.5，中英双语 SOTA 轻量模型）
    """

    backend: str = "auto"
    model: str = "BAAI/bge-small-zh-v1.5"

    @classmethod
    def from_env(cls) -> "EmbedConfig":
        return cls(
            backend=_get("EMBED_BACKEND", "auto"),
            model=_get("EMBED_MODEL", "BAAI/bge-small-zh-v1.5"),
        )


@dataclass
class IndexConfig:
    """向量索引配置。

    backend: ``auto`` | ``faiss`` | ``numpy``
    hnsw:    True 时 faiss 走 HNSW 近似索引（>10 万向量推荐）；默认 Flat 精确索引。
    """

    backend: str = "auto"
    hnsw: bool = False

    @classmethod
    def from_env(cls) -> "IndexConfig":
        return cls(
            backend=_get("INDEX_BACKEND", "auto"),
            hnsw=_get("INDEX_HNSW", "0") in ("1", "true", "True"),
        )


@dataclass
class RetrieveConfig:
    """检索配置。"""

    top_k: int = 20          # 混合检索每路候选数
    rerank_top_n: int = 4    # 重排后进入生成上下文的数量
    rrf_k: int = 60          # RRF 平滑常数（Cormack et al. 2009 推荐值）
    reranker_backend: str = "auto"  # auto | cross-encoder | cosine
    reranker_model: str = "BAAI/bge-reranker-base"

    @classmethod
    def from_env(cls) -> "RetrieveConfig":
        return cls(
            top_k=_get_int("TOP_K", 20),
            rerank_top_n=_get_int("RERANK_TOP_N", 4),
            rrf_k=_get_int("RRF_K", 60),
            reranker_backend=_get("RERANKER_BACKEND", "auto"),
            reranker_model=_get("RERANKER_MODEL", "BAAI/bge-reranker-base"),
        )


@dataclass
class LLMConfig:
    """LLM 配置。

    backend: ``mock``（确定性抽取式，测试/离线用）| ``openai``（任何
    OpenAI 兼容端点：llama.cpp / vLLM / DeepSeek / Qwen / Moonshot ...）
    """

    backend: str = "mock"
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    api_key: str = ""
    temperature: float = 0.1
    max_tokens: int = 1024

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            backend=_get("LLM_BACKEND", "mock"),
            model=_get("LLM_MODEL", "deepseek-chat"),
            base_url=_get("LLM_BASE_URL", "https://api.deepseek.com"),
            api_key=_get("LLM_API_KEY", ""),
            temperature=_get_float("LLM_TEMPERATURE", 0.1),
            max_tokens=_get_int("LLM_MAX_TOKENS", 1024),
        )


@dataclass
class VerifyConfig:
    """可验证性校验配置。

    min_coverage:      被检索证据支持的句子占比下限，低于则拒答
    support_threshold: 单句支持度阈值（嵌入余弦与词法 Jaccard 各半权重）
    """

    min_coverage: float = 0.6
    support_threshold: float = 0.35

    @classmethod
    def from_env(cls) -> "VerifyConfig":
        return cls(
            min_coverage=_get_float("MIN_COVERAGE", 0.6),
            support_threshold=_get_float("SUPPORT_THRESHOLD", 0.35),
        )


@dataclass
class QimingConfig:
    """总配置。"""

    chunk: ChunkConfig
    embed: EmbedConfig
    index: IndexConfig
    retrieve: RetrieveConfig
    llm: LLMConfig
    verify: VerifyConfig

    @classmethod
    def from_env(cls) -> "QimingConfig":
        return cls(
            chunk=ChunkConfig.from_env(),
            embed=EmbedConfig.from_env(),
            index=IndexConfig.from_env(),
            retrieve=RetrieveConfig.from_env(),
            llm=LLMConfig.from_env(),
            verify=VerifyConfig.from_env(),
        )

    def describe(self) -> dict:
        """返回当前配置快照（用于 /stats 与可复现记录）。"""
        return {
            "chunk": self.chunk.__dict__,
            "embed": self.embed.__dict__,
            "index": self.index.__dict__,
            "retrieve": self.retrieve.__dict__,
            "llm": {**self.llm.__dict__, "api_key": "***" if self.llm.api_key else ""},
            "verify": self.verify.__dict__,
        }
