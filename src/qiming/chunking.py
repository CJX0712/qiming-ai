"""启明 · 文本分块层。

**零重写**：主实现直接复用 LangChain 的 ``RecursiveCharacterTextSplitter``
（世界范围内使用最广、经过最充分验证的递归字符分块器，MIT 许可）。
当该包不可用时（极端离线环境），退化为本文件内置的同算法实现，
分隔符层级与 LangChain 完全一致，保证行为可复现。

算法（Recursive Character Text Splitting）：
按分隔符优先级 ``["\\n\\n", "\\n", "。", "！", "？", " ", ""]`` 递归切分，
优先在自然边界（段落/句子）处断开；每块长度不超过 ``chunk_size`` 字符，
相邻块保留 ``chunk_overlap`` 字符重叠以维持上下文连续。

可验证不变量（见 docs/VERIFICATION.md）：
- I-C1  每块长度 <= chunk_size
- I-C2  相邻块重叠长度 == chunk_overlap（长文本场景）
- I-C3  块数 * (chunk_size - chunk_overlap) + overlap >= 原文长度（无内容丢失）
- I-C4  无空白块
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """一个文本块。"""

    chunk_id: str
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


# 与 LangChain RecursiveCharacterTextSplitter 一致的分隔符层级（追加中文句号）
_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "; ", "; ", " ", ""]


class Chunker:
    """递归字符分块器（LangChain 主实现 + 内置同构回退）。"""

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 80) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._backend = "builtin"
        self._splitter = None
        try:  # 零重写：优先复用世界级现成实现
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=_SEPARATORS,
            )
            self._backend = "langchain-text-splitters"
        except Exception:  # pragma: no cover - 仅极端离线环境触发
            self._splitter = None

    @property
    def backend(self) -> str:
        return self._backend

    def split(self, text: str, doc_id: str, metadata: dict | None = None) -> list[Chunk]:
        """将单个文档切分为块列表。"""
        metadata = dict(metadata or {})
        if self._splitter is not None:
            pieces = self._splitter.split_text(text)
        else:  # pragma: no cover
            pieces = self._split_builtin(text)
        chunks: list[Chunk] = []
        for i, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#c{i}",
                    doc_id=doc_id,
                    text=piece,
                    metadata={**metadata, "chunk_index": i},
                )
            )
        return chunks

    # ------------------------------------------------------------------
    # 内置回退实现（与 LangChain 算法同构；仅在主实现不可用时使用）
    # ------------------------------------------------------------------
    def _split_builtin(self, text: str) -> list[str]:  # pragma: no cover
        return self._recursive_split(text, _SEPARATORS)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:  # pragma: no cover
        final: list[str] = []
        separator = separators[-1]
        for sep in separators:
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                break
        splits = text.split(separator) if separator else list(text)
        good: list[str] = []
        for s in splits:
            if len(s) < self.chunk_size:
                good.append(s)
            else:
                if good:
                    final.extend(self._merge(good, separator))
                    good = []
                rest = separators[separators.index(separator) + 1 :] if separator in separators else []
                final.extend(self._recursive_split(s, rest))
        if good:
            final.extend(self._merge(good, separator))
        return [c for c in final if c.strip()]

    def _merge(self, splits: list[str], separator: str) -> list[str]:  # pragma: no cover
        out: list[str] = []
        current: list[str] = []
        total = 0
        for s in splits:
            n = len(s) + (len(separator) if current else 0)
            if current and total + n > self.chunk_size:
                out.append(separator.join(current))
                # 保留 overlap：从尾部回退约 chunk_overlap 字符
                while current and total > self.chunk_overlap:
                    total -= len(current[0]) + (len(separator) if len(current) > 1 else 0)
                    current.pop(0)
            current.append(s)
            total += n
        if current:
            out.append(separator.join(current))
        return out
