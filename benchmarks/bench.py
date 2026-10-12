"""启明 · 基准评测（Benchmark）。

对黄金语料（data/corpus.jsonl，星尘协作平台 24 篇文档）与黄金问答集
（benchmarks/golden.jsonl，18 个可答 + 3 个应拒答）执行评测：

- **检索质量**：hit@1 / hit@4 / MRR（目标文档是否进入引用列表）
- **可验证性**：可答问题的有据回答率、应拒答问题的拒答率
- **延迟**：各阶段平均耗时（稠密/稀疏/融合/重排/生成/校验）

输出：终端 Markdown 表格 + benchmarks/results.json（可复现记录）。

运行::

    python -m benchmarks.bench                # 默认语料
    python -m benchmarks.bench --data PATH    # 指定语料目录
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from qiming.config import QimingConfig
from qiming.pipeline import Document, QimingPipeline

BENCH_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCH_DIR.parent / "data"


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_benchmark(data_dir: str | None = None, verbose: bool = True) -> dict:
    root = Path(data_dir) if data_dir else DATA_DIR
    corpus = load_jsonl(root / "corpus.jsonl")
    golden = load_jsonl(BENCH_DIR / "golden.jsonl")

    cfg = QimingConfig.from_env()
    pipe = QimingPipeline(cfg)
    t0 = time.perf_counter()
    pipe.ingest([Document(d["doc_id"], d["text"], {"title": d.get("title", "")}) for d in corpus])
    ingest_ms = (time.perf_counter() - t0) * 1000

    answerable = [g for g in golden if not g.get("expected_abstain")]
    unanswerable = [g for g in golden if g.get("expected_abstain")]

    # ---- 检索质量 ----
    hit1 = hit4 = 0
    mrr = 0.0
    for g in answerable:
        citations, _ = pipe.retrieve(g["question"])
        cited_docs = [c.chunk.doc_id for c in citations]
        rank = next((i + 1 for i, d in enumerate(cited_docs) if d in g["expected"]), None)
        if rank == 1:
            hit1 += 1
        if rank is not None and rank <= 4:
            hit4 += 1
        if rank is not None:
            mrr += 1.0 / rank
    n_ans = max(len(answerable), 1)

    # ---- 可验证性（全链路，MockLLM 保证可复现） ----
    answered_with_evidence = 0
    abstained_correctly = 0
    stage_totals: dict[str, float] = {}
    for g in answerable:
        ans = pipe.ask(g["question"])
        if not ans.abstained:
            answered_with_evidence += 1
        for k, v in ans.trace.items():
            stage_totals[k] = stage_totals.get(k, 0.0) + v
    for g in unanswerable:
        ans = pipe.ask(g["question"])
        if ans.abstained:
            abstained_correctly += 1

    n_un = max(len(unanswerable), 1)
    results = {
        "corpus_docs": len(corpus),
        "chunks": len(pipe.chunks),
        "backends": pipe.backends,
        "config": cfg.describe(),
        "ingest_ms": round(ingest_ms, 1),
        "retrieval": {
            "hit@1": round(hit1 / n_ans, 4),
            "hit@4": round(hit4 / n_ans, 4),
            "mrr": round(mrr / n_ans, 4),
            "n_answerable": len(answerable),
        },
        "verification": {
            "answered_with_evidence_rate": round(answered_with_evidence / n_ans, 4),
            "abstention_rate": round(abstained_correctly / n_un, 4),
            "n_unanswerable": len(unanswerable),
        },
        "latency_ms": {k: round(v / max(len(answerable), 1), 1) for k, v in stage_totals.items()},
    }

    out = BENCH_DIR / "results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    if verbose:
        _print_report(results)
    return results


def _print_report(r: dict) -> None:
    print("\n# 启明 Qiming · 基准评测报告\n")
    print(f"- 语料：{r['corpus_docs']} 篇 → {r['chunks']} 块，入库 {r['ingest_ms']:.0f} ms")
    print(f"- 后端：`{json.dumps(r['backends'], ensure_ascii=False)}`\n")
    print("## 检索质量\n")
    print("| 指标 | 数值 |")
    print("| --- | --- |")
    print(f"| hit@1 | **{r['retrieval']['hit@1']:.2%}** |")
    print(f"| hit@4 | **{r['retrieval']['hit@4']:.2%}** |")
    print(f"| MRR | **{r['retrieval']['mrr']:.3f}** |")
    print("\n## 可验证性\n")
    print("| 指标 | 数值 |")
    print("| --- | --- |")
    print(f"| 可答问题有据回答率 | **{r['verification']['answered_with_evidence_rate']:.2%}** |")
    print(f"| 应拒答问题拒答率 | **{r['verification']['abstention_rate']:.2%}** |")
    print("\n## 各阶段平均延迟（ms）\n")
    print("| 阶段 | 耗时 |")
    print("| --- | --- |")
    labels = {
        "dense_ms": "稠密召回（faiss）",
        "sparse_ms": "稀疏召回（BM25）",
        "fuse_ms": "RRF 融合",
        "rerank_ms": "Cross-Encoder 重排",
        "generate_ms": "生成（MockLLM）",
        "verify_ms": "可验证性校验",
        "total_ms": "端到端合计",
    }
    for k, v in r["latency_ms"].items():
        print(f"| {labels.get(k, k)} | {v:.1f} |")
    print(f"\n完整数据已写入 {BENCH_DIR / 'results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="启明基准评测")
    parser.add_argument("--data", default=None, help="黄金语料目录")
    args = parser.parse_args()
    run_benchmark(data_dir=args.data)
    sys.exit(0)
