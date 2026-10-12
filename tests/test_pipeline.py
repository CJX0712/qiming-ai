"""流水线端到端测试：hermetic 环境（确定性后端）下的全链路行为。"""

from __future__ import annotations

import json

from qiming.pipeline import Document


class TestPipeline:
    def test_ingest_stats(self, pipeline):
        stats = pipeline.ingest([Document("new-doc", "这是一篇全新的测试文档内容。")])
        assert stats["n_docs"] == 1
        assert stats["n_chunks"] >= 1

    def test_ask_known_question_answered_with_evidence(self, pipeline):
        ans = pipeline.ask("怎么重置登录密码？")
        assert not ans.abstained
        assert "重置" in ans.text
        assert ans.citations
        assert ans.verification.passed
        assert ans.citations[0].chunk.doc_id == "faq-pw"

    def test_ask_unknown_question_abstains(self, pipeline):
        ans = pipeline.ask("如何驾驶宇宙飞船？")
        assert ans.abstained
        assert ans.text == "根据现有资料，无法可靠回答该问题。"

    def test_citations_are_ranked_and_indexed(self, pipeline):
        ans = pipeline.ask("数据导出支持哪些格式？")
        indices = [c.index for c in ans.citations]
        assert indices == list(range(1, len(indices) + 1))
        scores = [c.score for c in ans.citations]
        assert all(a >= b for a, b in zip(scores, scores[1:]))

    def test_trace_has_all_stages(self, pipeline):
        ans = pipeline.ask("SSO 支持哪些协议？")
        for stage in ("dense_ms", "sparse_ms", "fuse_ms", "rerank_ms", "generate_ms", "verify_ms", "total_ms"):
            assert stage in ans.trace

    def test_retrieve_returns_citations_only(self, pipeline):
        citations, trace = pipeline.retrieve("审计日志保留多久？")
        assert all(c.chunk.chunk_id for c in citations)
        assert "dense_ms" in trace

    def test_save_and_load_roundtrip(self, pipeline, tmp_path):
        pipeline.save(tmp_path / "state")
        from qiming.config import QimingConfig
        from qiming.pipeline import QimingPipeline

        cfg = QimingConfig.from_env()
        cfg.embed.backend = "hashing"
        cfg.index.backend = "numpy"
        cfg.llm.backend = "mock"
        cfg.retrieve.reranker_backend = "cosine"
        restored = QimingPipeline.load(tmp_path / "state", cfg)
        assert len(restored.chunks) == len(pipeline.chunks)
        ans = restored.ask("怎么重置登录密码？")
        assert not ans.abstained
        assert "重置" in ans.text

    def test_backends_snapshot(self, pipeline):
        b = pipeline.backends
        assert "embedder" in b and "llm" in b
        assert "hashing" in b["embedder"]

    def test_empty_pipeline_ask_abstains(self, cfg):
        from qiming.pipeline import QimingPipeline

        pipe = QimingPipeline(cfg)
        ans = pipe.ask("任何问题")
        assert ans.abstained
