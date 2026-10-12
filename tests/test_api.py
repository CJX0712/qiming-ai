"""FastAPI 接口测试（TestClient 全链路）。"""

from __future__ import annotations

import pytest

from qiming.api import create_app
from qiming.config import QimingConfig
from qiming.pipeline import Document, QimingPipeline


def _hermetic_config() -> QimingConfig:
    cfg = QimingConfig.from_env()
    cfg.embed.backend = "hashing"
    cfg.index.backend = "numpy"
    cfg.llm.backend = "mock"
    cfg.retrieve.reranker_backend = "cosine"
    return cfg


@pytest.fixture(scope="module")
def client():
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    pipe = QimingPipeline(_hermetic_config())
    pipe.ingest(
        [
            Document("faq-pw", "用户可以在设置页面重置登录密码，重置链接会发送到注册邮箱。"),
            Document("faq-export", "数据导出支持 CSV 与 XLSX 格式，完成后会有站内通知。"),
        ]
    )
    return TestClient(create_app(pipe))


class TestAPI:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_ingest(self, client):
        r = client.post("/ingest", json={"documents": [{"doc_id": "api-doc", "text": "接口测试文档内容。"}]})
        assert r.status_code == 200
        assert r.json()["ingested"]["n_docs"] == 1

    def test_ask(self, client):
        r = client.post("/ask", json={"question": "怎么重置登录密码？"})
        assert r.status_code == 200
        body = r.json()
        assert body["abstained"] is False
        assert "重置" in body["answer"]
        assert body["citations"]
        assert body["verification"]["passed"] is True

    def test_ask_unknown_abstains(self, client):
        r = client.post("/ask", json={"question": "如何驾驶宇宙飞船？"})
        assert r.status_code == 200
        assert r.json()["abstained"] is True

    def test_stats(self, client):
        r = client.get("/stats")
        assert r.status_code == 200
        assert r.json()["n_chunks"] >= 2
        assert "embedder" in r.json()["backends"]

    def test_ask_empty_question_rejected(self, client):
        r = client.post("/ask", json={"question": ""})
        assert r.status_code == 422
