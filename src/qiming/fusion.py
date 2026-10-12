"""启明 · RRF 倒数排名融合层。

算法：Reciprocal Rank Fusion（Cormack, Clarke & Buettcher, SIGIR 2009，
被 TREC 大规模评测反复验证为最稳健的免训练融合方案）::

    RRF(d) = Σ_ranking  1 / (k + rank_ranking(d))

其中 ``k`` 为平滑常数，论文推荐 60。RRF 只用排名、不用原始分数，
因此稠密检索（余弦）与稀疏检索（BM25）两类异构得分无需归一化即可融合。

可验证不变量（见 docs/VERIFICATION.md）：
- I-F1  自融合：单路排名与自身 RRF 融合，顺序不变
- I-F2  一致性：任意一路中排第一的文档，融合分 >= 任一路第二的文档
- I-F3  对称性：交换两路顺序，融合结果完全一致
- I-F4  只出现于一路的文档，融合分 = 1/(k+rank)
"""

from __future__ import annotations

from typing import Sequence


def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """将多路排名（doc_id 序列，按相关度降序）融合为单一排名。

    参数
    ----
    rankings : 多路排名，每路为按相关度降序排列的 doc_id 列表
    k        : RRF 平滑常数，默认 60（Cormack et al. 2009 推荐值）

    返回
    ----
    ``[(doc_id, score), ...]`` 按融合分降序；score = Σ 1/(k + rank)
    """
    if k <= 0:
        raise ValueError("RRF 常数 k 必须为正整数")
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
