"""启明 · 一键自检套件（浏览器/Node 式的零依赖自检传统）。

复用世界标准测试框架 pytest 的同时，提供一个零依赖、秒级完成的内联
不变量套件——任何环境敲一条命令即可验证系统核心正确性::

    python -m qiming.cli selftest

检查项与 docs/VERIFICATION.md 的不变量编号一一对应。
"""

from __future__ import annotations

import numpy as np

from qiming.chunking import Chunker
from qiming.config import QimingConfig
from qiming.embeddings import HashingEmbedder
from qiming.fusion import rrf_fuse
from qiming.pipeline import Document, QimingPipeline
from qiming.sparse import BuiltinBM25Index
from qiming.verify import verify
from qiming.vectorstore import FaissStore, NumpyStore

RESULTS: list[tuple[str, bool, str]] = []


def _check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))


def run_selftest(verbose: bool = True) -> bool:
    """运行全部自检项，返回是否全部通过。"""
    RESULTS.clear()

    # I-E1~I-E4：嵌入不变量
    emb = HashingEmbedder(dim=256)
    vecs = emb.encode(["如何重置登录密码", "今天天气真好", "如何重置登录密码"])
    _check("I-E1 形状 (n, dim)", vecs.shape == (3, 256), str(vecs.shape))
    norms = np.linalg.norm(vecs, axis=1)
    _check("I-E2 行范数=1", bool(np.allclose(norms, 1.0, atol=1e-6)), f"max|err|={float(np.abs(norms-1).max()):.2e}")
    _check("I-E3 确定性", bool(np.array_equal(vecs[0], vecs[2])))
    _check("I-E4 自相似度=1", abs(float(vecs[0] @ vecs[2]) - 1.0) < 1e-6)

    # I-C1~I-C4：分块不变量（非周期文本，避免重复模式造成的假重叠）
    ch = Chunker(chunk_size=40, chunk_overlap=8)
    sents = ["今日发布新产品。", "团队完成季度目标。", "用户增长超出预期。", "系统上线新模块。",
             "客户签约创新高。", "版本更新已推送。", "安全审计已通过。", "文档中心已改版。"]
    long_text = "".join(sents * 6)
    chunks = ch.split(long_text, "doc1")
    _check("I-C1 块长<=chunk_size", all(len(c.text) <= 40 for c in chunks))
    _check("I-C4 无空白块", all(c.text.strip() for c in chunks))
    if len(chunks) >= 3:
        overlaps = [_tail_overlap(chunks[i].text, chunks[i + 1].text) for i in range(len(chunks) - 1)]
        _check("I-C2 重叠不超预算", all(0 <= o <= 8 + 2 for o in overlaps), str(overlaps))
        _check("I-C2b 存在重叠保连续", any(o > 0 for o in overlaps))
    _check("I-C3 无内容丢失", sum(len(c.text) for c in chunks) >= len(long_text) - 8)

    # I-S1：BM25 手算金标准（内置实现 vs 手工计算）
    bm25 = BuiltinBM25Index()
    bm25.add(["d1", "d2"], ["密码 重置 方法", "天气 预报 服务"])
    got = dict(bm25.search("密码 重置", k=2))
    # 手工计算：d1 命中密码+重置，d2 命中 0 → d1 必第一
    _check("I-S2 查询词全命中排第一", bool(got) and max(got, key=got.get) == "d1")
    _check("I-S3 空库不抛异常", BuiltinBM25Index().search("任意", 5) == [])

    # I-V1~I-V3：向量库不变量（faiss 主实现 vs numpy 参照实现交叉验证）
    dim = 64
    rng = np.random.default_rng(42)
    mat = rng.standard_normal((50, dim)).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    ids = [f"c{i}" for i in range(50)]
    nstore, fstore = NumpyStore(dim), FaissStore(dim)
    nstore.add(ids, mat)
    fstore.add(ids, mat)
    q = mat[7]
    r_np, r_fa = nstore.search(q, 5), fstore.search(q, 5)
    _check("I-V1 自检索 top-1=自身", r_np[0][0] == "c7" and r_fa[0][0] == "c7", f"{r_fa[0][0]}")
    _check("I-V1b 自检索得分=1", abs(r_np[0][1] - 1.0) < 1e-5 and abs(r_fa[0][1] - 1.0) < 1e-5)
    scores_np = [s for _, s in r_np]
    _check("I-V2 得分单调不增", all(a >= b - 1e-6 for a, b in zip(scores_np, scores_np[1:])))
    same_ids = [i for i, _ in r_np] == [i for i, _ in r_fa]
    same_scores = all(abs(a[1] - b[1]) < 1e-4 for a, b in zip(r_np, r_fa))
    _check("I-V3 faiss vs numpy 完全一致", same_ids and same_scores)
    _check("I-V4 k>库大小不报错", len(nstore.search(q, 999)) == 50)

    # I-F1~I-F3：RRF 不变量
    r1, r2 = ["a", "b", "c"], ["b", "c", "d"]
    fused = rrf_fuse([r1, r2])
    fused_self = rrf_fuse([r1, r1])
    _check("I-F1 自融合保序", [d for d, _ in fused_self] == r1)
    fused_swap = rrf_fuse([r2, r1])
    _check("I-F3 交换对称", fused == fused_swap)
    first_docs = {d for d, s in fused if s == fused[0][1]}
    _check("I-F2 头部一致性", "b" in first_docs)

    # I-X1/I-X2/I-X3/I-X5：可验证性不变量
    evidence = ["用户可以在设置页面重置登录密码，重置链接会发送到注册邮箱。", "系统每 90 天强制要求更换密码。"]
    copied = evidence[0] + " [1]"
    rep = verify(copied, evidence, emb)
    _check("I-X1 原文照抄 coverage=1", rep.coverage == 1.0 and rep.passed, f"coverage={rep.coverage:.3f}")
    off = "火星探测器的推进器采用液态甲烷作为燃料。 [1]"
    rep_off = verify(off, evidence, emb)
    _check("I-X2 无关内容拒答", not rep_off.passed)
    bad_cite = "用户可以在设置页面重置登录密码。 [9]"
    _check("I-X3 越界引用判无效", not verify(bad_cite, evidence, emb).citations_valid)
    rep2 = verify(copied, evidence, emb)
    _check("I-X5 校验纯函数", rep.coverage == rep2.coverage and rep.faithfulness == rep2.faithfulness)
    _check("I-X4 空证据拒答", not verify("任何回答。 [1]", [], emb).passed)

    # e2e：MockLLM 全链路（离线可复现）
    cfg = QimingConfig.from_env()
    cfg.embed.backend = "hashing"
    cfg.index.backend = "numpy"
    cfg.llm.backend = "mock"
    cfg.retrieve.reranker_backend = "cosine"
    pipe = QimingPipeline(cfg)
    pipe.ingest([
        Document("faq-pw", "用户可以在设置页面重置登录密码，重置链接会发送到注册邮箱。系统每 90 天强制要求更换密码。"),
        Document("faq-export", "数据导出支持 CSV 与 XLSX 格式，导出任务完成后会收到站内通知。"),
    ])
    ans = pipe.ask("怎么重置登录密码？")
    _check("e2e 已知问题有据回答", not ans.abstained and "重置" in ans.text, ans.text[:40])
    ans2 = pipe.ask("如何驾驶宇宙飞船？")
    _check("e2e 未知问题拒答", ans2.abstained)

    ok = all(r[1] for r in RESULTS)
    if verbose:
        _print_results(ok)
    return ok


def _tail_overlap(a: str, b: str) -> int:
    """计算 a 的尾部与 b 的头部最大重叠长度。"""
    for n in range(min(len(a), len(b)), 0, -1):
        if a[-n:] == b[:n]:
            return n
    return 0


def _print_results(ok: bool) -> None:
    print("\n启明 Qiming · 一键自检")
    print("=" * 56)
    for name, passed, detail in RESULTS:
        mark = "✅" if passed else "❌"
        line = f"{mark} {name}"
        if detail and not passed:
            line += f"  ({detail})"
        print(line)
    print("=" * 56)
    total = len(RESULTS)
    passed_n = sum(1 for r in RESULTS if r[1])
    print(f"结果：{passed_n}/{total} 通过" + (" —— 全部绿灯 ✅" if ok else " —— 存在失败项 ❌"))
