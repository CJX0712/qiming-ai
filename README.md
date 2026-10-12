# 启明 Qiming · 可验证的检索增强生成系统

> 天亮前最后还亮着的那颗星。这个系统的信条只有一条：**答必有据，据必可验**。
> 每一条回答都携带引用，每一次引用都经过可验证性校验，置信不足时宁可拒答，绝不编造。

[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-76%20passed-brightgreen.svg)](tests/)
[![Selftest](https://img.shields.io/badge/selftest-26%2F26-brightgreen.svg)](src/qiming/selftest.py)

**作者：晨星** · [GitHub: CJX0712](https://github.com/CJX0712)

---

## 1. 它解决什么问题

传统 RAG 的软肋：**检索到了资料，模型照样可能编造**。评估一个 RAG 系统"答得对不对"很难，
上线后"什么时候在胡说"更不知道。启明把可验证性做成一等公民：

| 传统 RAG | 启明 Qiming |
| --- | --- |
| 生成完就输出 | 生成后必须通过三层校验才输出 |
| 幻觉无法察觉 | 句级支持度逐句量化，覆盖率不足即拒答 |
| 引用真假不明 | 引用编号合法性 + 与最佳证据一致性双重检查 |
| 检索质量靠感觉 | hit@1 / hit@4 / MRR / 拒答率全量基准评测 |

## 2. 架构

```mermaid
flowchart LR
    subgraph Ingest["入库（可重复调用，增量追加）"]
        D[文档] --> C[分块<br/>RecursiveCharacterTextSplitter]
        C --> V[faiss 稠密索引<br/>IndexFlatIP / HNSW]
        C --> S[BM25 稀疏索引]
    end
    subgraph Ask["问答（全链路 trace）"]
        Q[问题] --> VQ[稠密召回 top-k]
        Q --> SQ[稀疏召回 top-k]
        VQ --> F[RRF 融合<br/>Cormack 2009]
        SQ --> F
        F --> R[Cross-Encoder 重排<br/>bge-reranker-base]
        R --> G[引用约束生成<br/>OpenAI 兼容 LLM]
        G --> X{可验证性校验<br/>句级支持度 + 引用有效性}
        X -->|通过| A[有据回答 + 引用]
        X -->|不通过| Z[拒答<br/>"根据现有资料，无法可靠回答该问题。"]
    end
```

**每一层都站在世界顶级开源成果之上（零重写原则）：**

| 层 | 采用的世界级方案 | 许可 |
| --- | --- | --- |
| 文本分块 | `langchain-text-splitters` RecursiveCharacterTextSplitter | MIT |
| 语义嵌入 | `fastembed` + `BAAI/bge-small-zh-v1.5`（C-MTEB 第一梯队） | MIT |
| 向量索引 | `faiss-cpu` IndexFlatIP / IndexHNSWFlat | MIT |
| 稀疏检索 | `rank_bm25` BM25Okapi | MIT |
| 排名融合 | Reciprocal Rank Fusion（Cormack, Clarke & Buettcher, SIGIR 2009） | 论文算法 |
| 精排 | `fastembed` TextCrossEncoder + `BAAI/bge-reranker-base` | MIT |
| LLM 接入 | OpenAI 官方 SDK（兼容 llama.cpp / vLLM / DeepSeek / Qwen…） | Apache-2.0 |
| HTTP 服务 | FastAPI + uvicorn | BSD |

**所有重依赖均可退化，系统在任何环境可运行、可复现、可验证：**

| 首选（SOTA） | 退化（确定性回退） | 退化后失去 | 退化后保留 |
| --- | --- | --- | --- |
| fastembed 语义嵌入 | 哈希嵌入（hashing trick） | 语义泛化 | 确定性、全部几何不变量 |
| faiss 向量索引 | numpy 暴力精确索引 | 大规模速度 | 完全精确、与主实现交叉一致 |
| cross-encoder 重排 | 双编码器余弦重排 | 精排精度 | 确定性排序 |
| OpenAI 兼容 LLM | MockLLM 抽取式 | 自然语言生成 | 端到端可复现基线 |

## 3. 快速开始

```bash
# 安装（Python 3.10+）
pip install -e ".[embed,index,api,dev]"

# 一键自检：26 项不变量，秒级完成，零网络
python -m qiming.cli selftest

# 测试套件
pytest tests/            # 76 passed

# 基准评测（黄金语料 24 篇 + 黄金问答 21 条）
python -m benchmarks.bench

# 入库 → 提问
python -m qiming.cli ingest data/corpus.jsonl --state state
python -m qiming.cli ask "如何重置登录密码？" --state state

# HTTP 服务
python -m qiming.cli serve --port 8000 --state state
curl -s localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "API 限流是多少？"}'
```

**接入真实 LLM**（任何 OpenAI 兼容端点）：

```bash
export QIMING_LLM_BACKEND=openai
export QIMING_LLM_BASE_URL=https://api.deepseek.com   # 或本地 llama.cpp: http://127.0.0.1:8080/v1
export QIMING_LLM_API_KEY=sk-...
export QIMING_LLM_MODEL=deepseek-chat
```

**国内网络下载模型**：`export HF_ENDPOINT=https://hf-mirror.com`

## 4. 基准评测（真实语义链路）

在 25 篇"星尘协作平台"文档语料上，使用完整 SOTA 链路
（bge-small-zh-v1.5 嵌入 + faiss 索引 + bge-reranker-base 重排 + BM25/RRF 混合），
18 个可答问题 + 3 个应拒答问题，实测结果：

| 指标 | 数值 |
| --- | --- |
| hit@1（目标文档排第一） | **94.44%** |
| hit@4（目标文档进引用列表） | **100.00%** |
| MRR | **0.972** |
| 可答问题有据回答率 | **100.00%** |
| 应拒答问题拒答率 | **100.00%** |

各阶段平均延迟（CPU，AMD Ryzen 7 H 255）：稠密召回 5.2 ms、BM25 0.4 ms、
RRF 0.0 ms、Cross-Encoder 重排 1215.7 ms（22 候选，CPU 瓶颈）、
MockLLM 生成 0.1 ms、可验证性校验 73.1 ms、端到端 1294.7 ms。

> 重排是端到端延迟的主要来源。调优路径：减小 `QIMING_TOP_K`、
> 换 int8 量化重排器（`BAAI/bge-reranker-v2-m3-int8`）、或启用
> ONNX Runtime 多线程。完整数据（含后端快照与全部配置）由
> `python -m benchmarks.bench` 生成并写入 `benchmarks/results.json`，
> 保证任何人在任何时间可复现。

## 5. 可验证性：三层校验 + 拒答

```
回答 ──► ① 句级支持度  每句与全部证据计算 0.5·cos + 0.5·Jaccard，取最大值
      ──► ② 引用有效性  [n] 必须 ∈ 1..len(证据)，且与句级最佳证据一致
      ──► ③ 覆盖率门槛  coverage ≥ 0.6 才放行，否则拒答
```

词法项保证"原文照抄"必然满分；语义项保证"同义转述"也能高分——
即使嵌入退化为哈希实现，校验器依然成立（这是测试套件完全 hermetic 的原因）。

## 6. 可验证不变量（硬金标准）

测试套件与 `python -m qiming.cli selftest` 对以下不变量做独立交叉验证
（参照实现 vs 主实现、解析解 vs 数值解、手算金标准 vs 实现）：

| 编号 | 不变量 | 验证方式 |
| --- | --- | --- |
| I-E1~E4 | 嵌入形状 / 单位范数 / 确定性 / 自相似度=1 | 直接断言 |
| I-E5 | 相关文本相似度 > 无关文本 | 语义模型可用时执行 |
| I-C1~C4 | 块长上限 / 重叠不超预算 / 无内容丢失 / 无空白块 | 直接断言 |
| I-S1 | BM25 得分 == Okapi 公式手算值（±1e-9） | 手算金标准 |
| I-S2/S3 | 全命中排第一 / 空库不抛异常 | 直接断言 |
| I-V1/V2 | 自检索 top-1=自身且得分=1 / 得分单调不增 | 直接断言 |
| I-V3 | **faiss 与 numpy 两个独立实现 top-k 完全一致** | 交叉验证 |
| I-V4 | k > 库大小不报错 | 边界测试 |
| I-F1~F4 | RRF 自融合保序 / 交换对称 / 头部一致 / 单路得分公式 | 数学断言 |
| I-R1~R3 | 重排保候选集 / 确定性 / 相关性排序 | 直接断言 |
| I-X1~X5 | 照抄 coverage=1 / 无关拒答 / 越界引用无效 / 空证据拒答 / 纯函数 | 直接断言 |

详见 [`docs/VERIFICATION.md`](docs/VERIFICATION.md)。

## 7. 项目结构

```
qiming-ai/
├── src/qiming/
│   ├── config.py       # 配置层（QIMING_* 环境变量全覆盖）
│   ├── chunking.py     # 分块（langchain 主实现 + 内置同构回退）
│   ├── embeddings.py   # 嵌入（fastembed 主实现 + 哈希回退）
│   ├── vectorstore.py  # 向量库（faiss 主实现 + numpy 参照实现）
│   ├── sparse.py       # BM25（rank_bm25 主实现 + 内置参照实现）
│   ├── fusion.py       # RRF 融合
│   ├── rerank.py       # Cross-Encoder 重排（+ 余弦回退）
│   ├── llm.py          # OpenAI 兼容 LLM + 确定性 MockLLM
│   ├── verify.py       # 可验证性校验（创新内核）
│   ├── pipeline.py     # 流水线编排（ingest/ask/save/load）
│   ├── api.py          # FastAPI 服务
│   ├── cli.py          # 命令行
│   └── selftest.py     # 一键自检（26 项不变量）
├── tests/              # 76 项测试（hermetic：零网络零模型）
├── benchmarks/         # 黄金语料评测 + bench.py
├── data/corpus.jsonl   # 25 篇示例语料
└── docs/               # 架构与验证文档
```

## 8. 设计哲学

1. **零重写**：不造轮子。每一层先检索世界已有最优解直接复用；回退实现只在
   主实现不可用时兜底，且与主实现同构、互为交叉验证。
2. **集大成**：嵌入、索引、稀疏、融合、重排、生成、服务，全部采用各自领域
   事实标准；两个独立实现互相验证是测试套件的核心方法。
3. **可验证**：26 项自检不变量 + 72 项测试 + 基准评测报告，全部可一键复现。
4. **拒答优先**：系统的目标不是"总能回答"，而是"只输出有据可查的回答"。

## 9. 路线图

- [ ] 嵌入模型可插拔（bge-m3 / gte / e5 系列）
- [ ] HNSW 近似索引的召回率-速度权衡基准
- [ ] NLI 蕴含模型作为第四层校验（交叉编码器判据）
- [ ] 流式生成 + 增量校验
- [ ] 多语言黄金评测集（C-MTEB 风格）

## 10. 许可与署名

Apache-2.0。作者：**晨星**（CJX0712）。

依赖的世界级开源项目版权归其各自作者，详见各项目许可。
