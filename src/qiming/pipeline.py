"""启明 · 流水线编排层。

ingest：文档 → 分块 → 双路索引（faiss 稠密 + BM25 稀疏）
ask  ：问题 → 双路召回 → RRF 融合 → Cross-Encoder 重排
       → 引用约束生成 → 可验证性校验 → （不通过则拒答）

全程 trace 记录各阶段耗时，基准评测与线上观测共用。
支持 save/load 持久化，保证实验可复现。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from qiming.chunking import Chunk, Chunker
from qiming.config import QimingConfig
from qiming.embeddings import get_embedder
from qiming.fusion import rrf_fuse
from qiming.llm import MockLLM, build_llm
from qiming.rerank import build_reranker
from qiming.sparse import build_sparse_index
from qiming.vectorstore import build_store
from qiming.verify import ABSTAIN_TEXT, VerificationReport, verify

logger = logging.getLogger("qiming.pipeline")


@dataclass
class Document:
    """待入库文档。"""

    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Citation:
    """一条引用。"""

    index: int  # 1-based
    chunk: Chunk
    score: float


@dataclass
class Answer:
    """一次问答的完整结果。"""

    question: str
    text: str
    abstained: bool
    citations: list[Citation]
    verification: VerificationReport
    trace: dict = field(default_factory=dict)


class QimingPipeline:
    """启明 RAG 流水线。"""

    def __init__(self, config: QimingConfig | None = None) -> None:
        self.config = config or QimingConfig.from_env()
        self.chunker = Chunker(
            chunk_size=self.config.chunk.chunk_size,
            chunk_overlap=self.config.chunk.chunk_overlap,
        )
        self.embedder = get_embedder(self.config.embed.backend, self.config.embed.model)
        self.store = build_store(
            backend=self.config.index.backend,
            dim=self.embedder.dim,
            hnsw=self.config.index.hnsw,
        )
        self.sparse = build_sparse_index()
        self.reranker = build_reranker(
            backend=self.config.retrieve.reranker_backend,
            model=self.config.retrieve.reranker_model,
            embedder=self.embedder,
        )
        self.llm = build_llm(
            backend=self.config.llm.backend,
            model=self.config.llm.model,
            base_url=self.config.llm.base_url,
            api_key=self.config.llm.api_key,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
        )
        self.chunks: list[Chunk] = []
        self._id2chunk: dict[str, Chunk] = {}

    # ------------------------------------------------------------------
    # 属性快照（可复现记录）
    # ------------------------------------------------------------------
    @property
    def backends(self) -> dict:
        return {
            "chunker": self.chunker.backend,
            "embedder": self.embedder.name,
            "vectorstore": getattr(self.store, "backend_name", self.store.backend),
            "sparse": self.sparse.backend,
            "reranker": self.reranker.name,
            "llm": self.llm.name,
        }

    # ------------------------------------------------------------------
    # ingest
    # ------------------------------------------------------------------
    def ingest(self, documents: list[Document]) -> dict:
        """文档入库：分块 → 双路索引（增量追加，可多次调用）。"""
        t0 = time.perf_counter()
        new_chunks: list[Chunk] = []
        for doc in documents:
            new_chunks.extend(self.chunker.split(doc.text, doc.doc_id, doc.metadata))
        if not new_chunks:
            return {"n_docs": len(documents), "n_chunks": len(self.chunks), "elapsed_ms": 0.0}

        vectors = self.embedder.encode([c.text for c in new_chunks])
        ids = [c.chunk_id for c in new_chunks]
        self.store.add(ids, vectors)
        self.sparse.add(ids, [c.text for c in new_chunks])
        self.chunks.extend(new_chunks)
        self._id2chunk.update({c.chunk_id: c for c in new_chunks})
        return {
            "n_docs": len(documents),
            "n_chunks": len(self.chunks),
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
        }

    # ------------------------------------------------------------------
    # ask
    # ------------------------------------------------------------------
    def retrieve(self, question: str) -> list[Citation]:
        """混合检索：双路召回 → RRF 融合 → 重排。"""
        trace: dict = {}

        t0 = time.perf_counter()
        qv = self.embedder.encode([question])[0]
        dense_hits = self.store.search(qv, self.config.retrieve.top_k)
        trace["dense_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        sparse_hits = self.sparse.search(question, self.config.retrieve.top_k)
        trace["sparse_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        fused = rrf_fuse(
            [[cid for cid, _ in dense_hits], [cid for cid, _ in sparse_hits]],
            k=self.config.retrieve.rrf_k,
        )
        trace["fuse_ms"] = (time.perf_counter() - t0) * 1000

        candidate_ids = [cid for cid, _ in fused]
        candidate_chunks = [self._id2chunk[cid] for cid in candidate_ids if cid in self._id2chunk]

        t0 = time.perf_counter()
        scores = self.reranker.score(question, [c.text for c in candidate_chunks])
        trace["rerank_ms"] = (time.perf_counter() - t0) * 1000

        ranked = sorted(zip(candidate_chunks, scores), key=lambda x: -x[1])
        top = ranked[: self.config.retrieve.rerank_top_n]
        trace["n_candidates"] = len(candidate_chunks)
        return [
            Citation(index=i + 1, chunk=chunk, score=float(score))
            for i, (chunk, score) in enumerate(top)
        ], trace

    def ask(self, question: str) -> Answer:
        """完整问答：检索 → 生成 → 校验 → 拒答兜底。"""
        citations, trace = self.retrieve(question)
        contexts = [c.chunk.text for c in citations]

        t0 = time.perf_counter()
        draft = self.llm.generate(question, contexts)
        trace["generate_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        report = verify(
            draft,
            contexts,
            self.embedder,
            min_coverage=self.config.verify.min_coverage,
            support_threshold=self.config.verify.support_threshold,
        )
        trace["verify_ms"] = (time.perf_counter() - t0) * 1000

        if report.passed:
            text = draft
            abstained = False
        else:
            text = ABSTAIN_TEXT
            abstained = True
        trace["total_ms"] = sum(trace.get(k, 0.0) for k in ("dense_ms", "sparse_ms", "fuse_ms", "rerank_ms", "generate_ms", "verify_ms"))
        return Answer(
            question=question,
            text=text,
            abstained=abstained,
            citations=citations,
            verification=report,
            trace=trace,
        )

    # ------------------------------------------------------------------
    # 持久化（可复现）
    # ------------------------------------------------------------------
    def save(self, state_dir: str | Path) -> None:
        """保存索引状态：向量 npy + 块 jsonl + 配置 json。"""
        state_dir = Path(state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        vecs = self.embedder.encode([c.text for c in self.chunks]) if self.chunks else np.zeros((0, self.embedder.dim), np.float32)
        np.save(state_dir / "vectors.npy", vecs)
        with open(state_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps({"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text, "metadata": c.metadata}, ensure_ascii=False) + "\n")
        with open(state_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump({"config": self.config.describe(), "backends": self.backends}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, state_dir: str | Path, config: QimingConfig | None = None) -> "QimingPipeline":
        """从磁盘恢复流水线（无需重新嵌入，直接读回向量）。"""
        state_dir = Path(state_dir)
        pipeline = cls(config)
        with open(state_dir / "chunks.jsonl", encoding="utf-8") as f:
            pipeline.chunks = [
                Chunk(chunk_id=r["chunk_id"], doc_id=r["doc_id"], text=r["text"], metadata=r["metadata"])
                for r in map(json.loads, f)
            ]
        pipeline._id2chunk = {c.chunk_id: c for c in pipeline.chunks}
        vecs = np.load(state_dir / "vectors.npy")
        if len(pipeline.chunks):
            pipeline.store.add([c.chunk_id for c in pipeline.chunks], vecs)
            pipeline.sparse.add([c.chunk_id for c in pipeline.chunks], [c.text for c in pipeline.chunks])
        return pipeline
