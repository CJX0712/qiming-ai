"""启明 Qiming —— 可验证的检索增强生成系统（Verifiable RAG）。

天亮前最后还亮着的那颗星。这个系统的信条只有一条：
**答必有据，据必可验**——每一条回答都携带引用，每一次引用都经过
可验证性校验，置信不足时宁可拒答，绝不编造。

模块地图::

    qiming.config       配置层（环境变量 QIMING_* 全覆盖）
    qiming.chunking     文本分块（langchain RecursiveCharacterTextSplitter）
    qiming.embeddings   向量嵌入（fastembed/ONNX，哈希回退）
    qiming.vectorstore  向量索引（faiss 主实现 / numpy 参照实现）
    qiming.sparse       BM25 稀疏索引（rank_bm25）
    qiming.fusion       RRF 倒数排名融合（Cormack et al. 2009）
    qiming.rerank       Cross-Encoder 重排（fastembed，余弦回退）
    qiming.llm          OpenAI 兼容 LLM 客户端 + 确定性 MockLLM
    qiming.verify       可验证性校验（句级支持度 / 引用有效性 / 拒答）
    qiming.pipeline     流水线编排（ingest / ask / save / load）
    qiming.api          FastAPI HTTP 服务
    qiming.cli          命令行（selftest / ingest / ask / serve / bench）
    qiming.selftest     一键自检不变量套件

作者：晨星
许可：Apache-2.0
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "晨星"

from qiming.config import QimingConfig  # noqa: E402

__all__ = ["QimingConfig", "__version__", "__author__"]
