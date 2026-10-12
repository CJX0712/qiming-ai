"""启明 · 可验证性校验层（系统的创新内核）。

传统 RAG 的软肋：检索到了资料，模型照样可能编造。启明的答案：
**不信任生成，只信任证据**。每一次回答都必须经过三层校验——

1. **句级支持度**（sentence-level support）
   回答被切分为句子；每句与全部候选证据计算支持度
   ``support = 0.5·cos(embedding) + 0.5·Jaccard(tokens)``，
   取所有证据中的最大值。词法项保证"原文照抄"必然满分（即使
   嵌入退化为哈希实现），语义项保证"同义转述"也能高分。

2. **引用有效性**（citation validity）
   回答中标注的 ``[n]`` 必须落在 1..len(证据) 范围内，
   且被引证据必须真实支持至少一个句子（支持度 ≥ 阈值）——
   允许引用次优证据，但绝不放过"张冠李戴"。

3. **覆盖率门槛 + 拒答**（coverage gate & abstention）
   支持度达标的句子占比 ``coverage`` 低于 ``min_coverage``
   时，系统**拒绝输出该回答**，返回标准拒答话术——
   宁可说"无法回答"，绝不输出无据可查的内容。

可验证不变量（见 docs/VERIFICATION.md）：
- I-X1  回答逐字摘自某证据（完整子集）→ coverage = 1.0；
        每句均与某证据逐字相同 → faithfulness = 1.0
- I-X2  回答与证据完全无关 → coverage ≈ 0, 拒答
- I-X3  引用编号越界 → citations_valid = False
- I-X4  空证据 → 直接拒答（不进入生成后的校验）
- I-X5  校验器为纯函数：同输入必同输出（可复现）
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from qiming.chunking import Chunk
from qiming.embeddings import tokenize
from qiming.textutil import split_sentences

ABSTAIN_TEXT = "根据现有资料，无法可靠回答该问题。"

_CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class SentenceSupport:
    """单句的校验结果。"""

    sentence: str
    support: float
    best_evidence: int | None  # 最佳证据序号（1-based），None 表示无支持


@dataclass
class VerificationReport:
    """一次回答的完整校验报告。"""

    coverage: float                      # 支持度达标的句子占比
    faithfulness: float                  # 平均句级支持度
    citations_valid: bool                # 引用编号是否全部合法且一致
    cited_indices: list[int] = field(default_factory=list)
    sentences: list[SentenceSupport] = field(default_factory=list)
    passed: bool = False

    def summary(self) -> str:
        status = "✅ 通过" if self.passed else "❌ 拒答"
        return (
            f"{status} | coverage={self.coverage:.2f} "
            f"faithfulness={self.faithfulness:.2f} citations_valid={self.citations_valid}"
        )


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def verify(
    answer: str,
    contexts: Sequence[str],
    embedder,
    min_coverage: float = 0.6,
    support_threshold: float = 0.35,
) -> VerificationReport:
    """校验一次回答的证据支持度。

    参数
    ----
    answer            : 待校验的回答文本
    contexts          : 检索到的证据片段（按相关度降序）
    embedder          : 嵌入器（提供语义余弦项）
    min_coverage      : 覆盖率下限
    support_threshold : 单句支持度达标阈值

    返回
    ----
    VerificationReport（纯函数，无副作用）
    """
    sentences = split_sentences(_CITATION_RE.sub("", answer))  # 引用标记是标注而非内容，先剥离
    if not sentences:
        return VerificationReport(
            coverage=0.0,
            faithfulness=0.0,
            citations_valid=False,
            passed=False,
        )

    if not contexts:
        # I-X4：无证据直接拒答
        return VerificationReport(
            coverage=0.0,
            faithfulness=0.0,
            citations_valid=False,
            sentences=[SentenceSupport(s, 0.0, None) for s in sentences],
            passed=False,
        )

    # 批量编码：句子 + 证据，一次前向
    texts = sentences + list(contexts)
    vecs = embedder.encode(texts)
    sent_vecs = vecs[: len(sentences)]
    ctx_vecs = vecs[len(sentences) :]

    ctx_tokens = [set(tokenize(c)) for c in contexts]
    n_ctx = len(contexts)

    # 支持度矩阵：sentences × contexts
    matrix = np.zeros((len(sentences), n_ctx), dtype=np.float64)
    for i, sent in enumerate(sentences):
        sent_tok = set(tokenize(sent))
        for j in range(n_ctx):
            cos = float(np.dot(sent_vecs[i], ctx_vecs[j]))
            jac = _jaccard(sent_tok, ctx_tokens[j])
            matrix[i, j] = 0.5 * cos + 0.5 * jac

    supports: list[SentenceSupport] = []
    for i, sent in enumerate(sentences):
        best_idx = int(np.argmax(matrix[i]))
        supports.append(
            SentenceSupport(
                sentence=sent,
                support=float(matrix[i, best_idx]),
                best_evidence=best_idx + 1,  # 1-based
            )
        )

    n = len(sentences)
    covered = sum(1 for s in supports if s.support >= support_threshold)
    coverage = covered / n
    faithfulness = sum(s.support for s in supports) / n

    # 引用校验（grounded citation）：
    # ① 编号必须落在 1..len(证据) 范围内；
    # ② 被引证据必须真实支持至少一个句子（该句对其支持度 ≥ 阈值）——
    #    允许引用次优证据（真实 LLM 常引用非 argmax 的合理证据），
    #    但绝不放过"张冠李戴"（被引证据对所有句子均无支持）。
    cited_indices = [int(m.group(1)) for m in _CITATION_RE.finditer(answer)]
    citations_valid = True
    for idx in cited_indices:
        if idx < 1 or idx > n_ctx:
            citations_valid = False
            break
        if float(matrix[:, idx - 1].max()) < support_threshold:
            citations_valid = False
            break

    passed = bool(citations_valid and coverage >= min_coverage)
    return VerificationReport(
        coverage=coverage,
        faithfulness=faithfulness,
        citations_valid=citations_valid,
        cited_indices=cited_indices,
        sentences=supports,
        passed=passed,
    )
