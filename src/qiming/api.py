"""启明 · HTTP API 层（FastAPI）。

端点：
- ``GET  /health``   存活探针
- ``POST /ingest``   文档入库 ``{"documents": [{"doc_id","text","metadata"?}]}``
- ``POST /ask``      问答 ``{"question": "..."}`` → 回答 + 引用 + 校验报告
- ``GET  /stats``    索引规模与各层后端快照
"""

from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI
from pydantic import BaseModel, Field

from qiming.config import QimingConfig
from qiming.pipeline import Document, QimingPipeline


class DocumentIn(BaseModel):
    doc_id: str
    text: str
    metadata: dict = Field(default_factory=dict)


class IngestIn(BaseModel):
    documents: list[DocumentIn]


class AskIn(BaseModel):
    question: str = Field(min_length=1)


class CitationOut(BaseModel):
    index: int
    chunk_id: str
    doc_id: str
    text: str
    score: float


class AskOut(BaseModel):
    question: str
    answer: str
    abstained: bool
    citations: list[CitationOut]
    verification: dict
    trace: dict


def create_app(pipeline: QimingPipeline | None = None, config: QimingConfig | None = None) -> FastAPI:
    """应用工厂（依赖注入，便于测试）。"""
    app = FastAPI(title="启明 Qiming · Verifiable RAG", version="0.1.0")
    pipe = pipeline or QimingPipeline(config)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": "qiming", "version": "0.1.0"}

    @app.post("/ingest")
    def ingest(body: IngestIn) -> dict:
        stats = pipe.ingest(
            [Document(doc_id=d.doc_id, text=d.text, metadata=d.metadata) for d in body.documents]
        )
        return {"ingested": stats, "backends": pipe.backends}

    @app.post("/ask", response_model=AskOut)
    def ask(body: AskIn) -> AskOut:
        result = pipe.ask(body.question)
        return AskOut(
            question=result.question,
            answer=result.text,
            abstained=result.abstained,
            citations=[
                CitationOut(
                    index=c.index,
                    chunk_id=c.chunk.chunk_id,
                    doc_id=c.chunk.doc_id,
                    text=c.chunk.text,
                    score=c.score,
                )
                for c in result.citations
            ],
            verification={
                "coverage": result.verification.coverage,
                "faithfulness": result.verification.faithfulness,
                "citations_valid": result.verification.citations_valid,
                "passed": result.verification.passed,
            },
            trace=result.trace,
        )

    @app.get("/stats")
    def stats() -> dict:
        return {
            "n_chunks": len(pipe.chunks),
            "n_docs": len({c.doc_id for c in pipe.chunks}),
            "backends": pipe.backends,
            "config": pipe.config.describe(),
        }

    return app


app = create_app()
