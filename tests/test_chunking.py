"""分块层不变量测试（I-C1 ~ I-C4）。

注意：I-C2 的表述为"重叠不超过配置预算（含分隔符余量）"而非"恒等于
chunk_overlap"——LangChain 的合并策略在自然边界处丢弃不足一个 split 的
尾部，因此实际重叠 ∈ [0, chunk_overlap + max_separator]。这是对
RecursiveCharacterTextSplitter 真实行为的精确刻画。
"""

from __future__ import annotations

import pytest

from qiming.chunking import Chunker

# 非周期文本：周期文本会让"最长公共后缀/前缀"测量失真（重复模式的假重叠）
SENTENCES = [
    "今日发布新产品。",
    "团队完成季度目标。",
    "用户增长超出预期。",
    "系统上线新模块。",
    "客户签约创新高。",
    "版本更新已推送。",
    "安全审计已通过。",
    "文档中心已改版。",
]
TEXT = "".join(SENTENCES * 6)


def _tail_overlap(a: str, b: str) -> int:
    for n in range(min(len(a), len(b)), 0, -1):
        if a[-n:] == b[:n]:
            return n
    return 0


class TestChunker:
    def test_i_c1_chunk_size_respected(self):
        ch = Chunker(chunk_size=40, chunk_overlap=8)
        chunks = ch.split(TEXT, "d1")
        assert chunks
        assert all(len(c.text) <= 40 for c in chunks)

    def test_i_c2_overlap_within_budget(self):
        ch = Chunker(chunk_size=40, chunk_overlap=8)
        chunks = ch.split(TEXT, "d1")
        assert len(chunks) >= 3
        overlaps = [_tail_overlap(chunks[i].text, chunks[i + 1].text) for i in range(len(chunks) - 1)]
        assert all(0 <= o <= 8 + 2 for o in overlaps), overlaps
        assert any(o > 0 for o in overlaps), "长文本应至少存在一处重叠以维持上下文连续"

    def test_i_c3_no_content_loss(self):
        ch = Chunker(chunk_size=40, chunk_overlap=8)
        chunks = ch.split(TEXT, "d1")
        assert sum(len(c.text) for c in chunks) >= len(TEXT) - 8

    def test_i_c4_no_empty_chunks(self):
        ch = Chunker(chunk_size=20, chunk_overlap=5)
        chunks = ch.split("  内容  \n\n  " * 10, "d1")
        assert all(c.text.strip() for c in chunks)

    def test_chunk_ids_unique_and_ordered(self):
        ch = Chunker(chunk_size=40, chunk_overlap=8)
        chunks = ch.split(TEXT, "doc-7")
        ids = [c.chunk_id for c in chunks]
        assert len(set(ids)) == len(ids)
        assert all(i.startswith("doc-7#c") for i in ids)

    def test_short_text_single_chunk(self):
        ch = Chunker(chunk_size=100, chunk_overlap=10)
        chunks = ch.split("这是一段很短的内容。", "d1")
        assert len(chunks) == 1

    def test_overlap_must_be_smaller_than_size(self):
        with pytest.raises(ValueError):
            Chunker(chunk_size=10, chunk_overlap=10)

    def test_backend_is_langchain_or_builtin(self):
        ch = Chunker()
        assert ch.backend in ("langchain-text-splitters", "builtin")
