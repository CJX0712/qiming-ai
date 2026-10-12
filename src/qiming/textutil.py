"""启明 · 文本工具（分句）。

分句规则：中文句末标点（。！？）恒为边界；英文句点 ``.`` 仅在其后为
空白或字符串结尾时才算边界——**版本号/小数（v3.2、SAML 2.0、99.9）
中的句点不是句子边界**。这是可验证性校验与 MockLLM 共用的基础组件。
"""

from __future__ import annotations

import re

# 零宽断言：中文标点后 / 英文句点后接空白或结尾
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?])|(?<=\.)(?=\s)|(?<=\.)$")


def split_sentences(text: str) -> list[str]:
    """将文本切分为句子列表（保留句末标点，去除空白句）。"""
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
