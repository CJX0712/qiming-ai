"""启明 · 命令行入口。

子命令：
- ``selftest``   一键自检（零依赖不变量套件，秒级完成）
- ``ingest``     文档入库（txt/md/jsonl）并保存索引状态
- ``ask``        提问（加载已有索引状态）
- ``serve``      启动 FastAPI 服务
- ``bench``      运行基准评测
- ``version``    版本与后端信息
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qiming", description="启明 Qiming · 可验证 RAG 系统")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selftest", help="一键自检")

    p_ing = sub.add_parser("ingest", help="文档入库")
    p_ing.add_argument("files", nargs="+", help="txt/md/jsonl 文件路径")
    p_ing.add_argument("--state", default="state", help="索引状态目录")

    p_ask = sub.add_parser("ask", help="提问")
    p_ask.add_argument("question", help="问题文本")
    p_ask.add_argument("--state", default="state", help="索引状态目录")

    p_srv = sub.add_parser("serve", help="启动 API 服务")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8000)
    p_srv.add_argument("--state", default="state", help="索引状态目录")

    p_bench = sub.add_parser("bench", help="基准评测")
    p_bench.add_argument("--data", default=None, help="黄金语料目录（默认包内 data/）")

    sub.add_parser("version", help="版本与后端信息")

    args = parser.parse_args(argv)

    if args.cmd == "selftest":
        from qiming.selftest import run_selftest

        return 0 if run_selftest() else 1

    if args.cmd == "version":
        import qiming

        print(f"qiming-ai {qiming.__version__} | 作者：{qiming.__author__}")
        return 0

    if args.cmd == "ingest":
        from qiming.config import QimingConfig
        from qiming.pipeline import Document, QimingPipeline

        docs: list[Document] = []
        for fp in args.files:
            path = Path(fp)
            if path.suffix == ".jsonl":
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        r = json.loads(line)
                        docs.append(Document(r["doc_id"], r["text"], r.get("metadata", {})))
            else:
                docs.append(Document(path.stem, path.read_text(encoding="utf-8")))
        pipe = QimingPipeline(QimingConfig.from_env())
        stats = pipe.ingest(docs)
        pipe.save(args.state)
        print(f"入库完成：{stats['n_docs']} 文档 → {stats['n_chunks']} 块（{stats['elapsed_ms']:.0f} ms）")
        print(f"后端快照：{json.dumps(pipe.backends, ensure_ascii=False)}")
        return 0

    if args.cmd == "ask":
        from qiming.config import QimingConfig
        from qiming.pipeline import QimingPipeline

        pipe = QimingPipeline.load(args.state, QimingConfig.from_env())
        ans = pipe.ask(args.question)
        print(f"\n问：{ans.question}")
        print(f"答：{ans.text}")
        print(f"\n{ans.verification.summary()}")
        for c in ans.citations:
            print(f"  [{c.index}] {c.chunk.doc_id}  score={c.score:.3f}  {c.chunk.text[:60]}…")
        print(f"\ntrace: " + json.dumps({k: round(v, 1) for k, v in ans.trace.items()}, ensure_ascii=False))
        return 0

    if args.cmd == "serve":
        import uvicorn

        from qiming.api import create_app
        from qiming.config import QimingConfig
        from qiming.pipeline import QimingPipeline

        state = Path(args.state)
        if state.exists():
            pipe = QimingPipeline.load(state, QimingConfig.from_env())
            print(f"已加载索引状态：{state}")
        else:
            pipe = QimingPipeline(QimingConfig.from_env())
            print("未找到索引状态，以空索引启动")
        uvicorn.run(create_app(pipe), host=args.host, port=args.port)
        return 0

    if args.cmd == "bench":
        from benchmarks.bench import run_benchmark

        run_benchmark(data_dir=args.data)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
