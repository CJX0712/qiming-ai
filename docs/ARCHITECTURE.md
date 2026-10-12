# 启明 Qiming · 架构文档

## 1. 总览

启明是一个**可验证的检索增强生成（Verifiable RAG）**系统。核心主张：
RAG 的可信度不应来自"相信模型"，而应来自"证据可校验"。

```
                 ┌──────────────────────────────────────────────┐
                 │                  QimingPipeline               │
                 │                                              │
   文档 ──► 分块 ──► ┌────────────┐    ┌────────────┐            │
                 │   │ faiss 稠密  │    │ BM25 稀疏   │            │
                 │   │ 索引(主实现) │    │ 索引(主实现) │            │
                 │   └─────┬──────┘    └─────┬──────┘            │
                 │         └────── RRF ──────┘                   │
                 │                  │                            │
   问题 ──► 问题编码 ─────────► 候选融合 ──► Cross-Encoder 重排     │
                 │                  │                            │
                 │                  ▼                            │
                 │           引用约束生成（LLM）                   │
                 │                  │                            │
                 │                  ▼                            │
                 │     ┌─────────────────────────┐               │
                 │     │  可验证性校验（三层）      │               │
                 │     │  ① 句级支持度             │               │
                 │     │  ② 引用有效性             │               │
                 │     │  ③ 覆盖率门槛             │               │
                 │     └────────┬────────────────┘               │
                 │        通过  │  不通过                         │
                 │              ▼        ▼                       │
                 │        有据回答    拒答（ Abstain ）            │
                 └──────────────────────────────────────────────┘
```

## 2. 分层设计与选型（零重写 / 集大成）

每一层遵循同一选型纪律：**先检索世界已有最优解，直接复用；只在主实现
不可用时退化到同构回退，并与主实现互为交叉验证。**

### 2.1 分块层 `chunking.py`

- 主实现：LangChain `RecursiveCharacterTextSplitter`
- 算法：按分隔符优先级 `["\n\n", "\n", "。", "！", "？", "; ", " ", ""]` 递归切分，
  优先在自然边界断开；块长 ≤ `chunk_size`，相邻块保留 ≤ `chunk_overlap` 的重叠。
- 回退：内置同算法实现（仅 langchain 包缺失时启用）。

### 2.2 嵌入层 `embeddings.py`

- 主实现：`fastembed`（Qdrant，ONNX Runtime，无需 GPU/torch），默认
  `BAAI/bge-small-zh-v1.5`（512 维，C-MTEB 中文检索第一梯队轻量模型）。
- 回退：**确定性哈希嵌入**——blake2b(token) → 维度+符号，累加后 L2 归一化。
  数学上是 one-hot 稀疏随机投影的确定性实例，无需训练与网络。
- 契约：输出 `(n, dim)`、每行单位范数（内积即余弦）、同输入同输出。

### 2.3 向量索引层 `vectorstore.py`

- 主实现：`faiss` `IndexFlatIP`（精确）或 `IndexHNSWFlat`（近似，M=32）。
- 参照实现：numpy 暴力 `V @ q` + argpartition。
- **两个独立实现互为交叉验证**（测试 I-V3）：对同一查询 top-k 完全一致。

### 2.4 稀疏检索层 `sparse.py`

- 主实现：`rank_bm25` BM25Okapi。
- 参照实现：内置 Okapi 公式直译（Lucene 式非负 idf：`ln(1 + (N-df+0.5)/(df+0.5))`）。
- 交叉验证方式：排名一致性（测试 I-S 系列）；得分金标准用手算值核对内置实现。

### 2.5 融合层 `fusion.py`

Reciprocal Rank Fusion（Cormack, Clarke & Buettcher, SIGIR 2009）：

```
RRF(d) = Σ_rankings 1 / (k + rank(d))     k = 60
```

只用排名不用原始分数 → 余弦与 BM25 两类异构得分无需归一化即可融合。

### 2.6 重排层 `rerank.py`

- 主实现：fastembed `TextCrossEncoder`（`BAAI/bge-reranker-base`，ONNX）。
- 回退：双编码器余弦（直接用嵌入层打分）。
- retrieve-then-rerank 两阶段范式：混合检索保 recall，cross-encoder 提 precision。

### 2.7 生成层 `llm.py`

- `OpenAILLM`：OpenAI 官方 SDK，对接任何 OpenAI 兼容端点
  （llama.cpp / vLLM / Ollama / DeepSeek / 通义 / Moonshot）。
- `MockLLM`：确定性抽取式生成。**模拟遵循指令的 LLM**：问题与最相关
  资料无词法重合时回答"无法回答"（对应真实 LLM 在系统提示词约束下的
  拒答行为），否则摘录首两句并标注 `[1]`。测试与基准因此完全 hermetic。
- 提示词策略：**引用约束生成**——逐句标注 `[n]`，资料不足必须回答"无法回答"。

### 2.8 可验证性层 `verify.py`（创新内核）

三层校验，任一不过即拒答：

| 层 | 机制 | 数学 |
| --- | --- | --- |
| ① 句级支持度 | 每句 vs 全部证据取最大 | `support = 0.5·cos(embed) + 0.5·Jaccard(tokens)` |
| ② 引用有效性 | `[n]` 范围合法 + 与最佳证据一致 | 正则 + 索引比对 |
| ③ 覆盖率门槛 | `coverage = 达标句数 / 总句数 ≥ 0.6` | 布尔放行 |

词法项保证"原文照抄"满分；语义项保证"同义转述"高分。两项混合使校验器
在语义嵌入与哈希回退下都成立——这是 hermetic 测试的前提。

### 2.9 编排层 `pipeline.py`

`ingest`（增量追加，可多次调用）/ `ask`（全链路 trace：稠密、稀疏、融合、
重排、生成、校验各阶段毫秒数）/ `save` / `load`（磁盘状态可复现）。

## 3. 数据流时序

```
ingest(documents):
  for doc in documents: chunks += chunker.split(doc)
  vectors = embedder.encode(chunks)          # 批量前向
  faiss.add(ids, vectors); bm25.add(ids, texts)

ask(question):
  qv        = embedder.encode([question])
  dense     = faiss.search(qv, top_k=20)
  sparse    = bm25.search(question, top_k=20)
  fused     = rrf_fuse([dense_ids, sparse_ids], k=60)
  rerank    = cross_encoder.score(question, candidates)
  contexts  = top rerank_top_n=4 chunks
  draft     = llm.generate(question, contexts)
  report    = verify(draft, contexts, embedder)
  return    = draft if report.passed else ABSTAIN_TEXT
```

## 4. 配置矩阵

全部参数经 `QIMING_*` 环境变量覆盖（见 `config.py`）。关键旋钮：

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `QIMING_EMBED_BACKEND` | auto | auto / fastembed / hashing |
| `QIMING_INDEX_BACKEND` | auto | auto / faiss / numpy |
| `QIMING_INDEX_HNSW` | 0 | 1 = HNSW 近似索引（>10 万向量） |
| `QIMING_TOP_K` | 20 | 混合检索每路候选数 |
| `QIMING_RERANK_TOP_N` | 4 | 进入生成的上下文数 |
| `QIMING_RRF_K` | 60 | RRF 平滑常数 |
| `QIMING_LLM_BACKEND` | mock | mock / openai |
| `QIMING_MIN_COVERAGE` | 0.6 | 拒答门槛 |
| `QIMING_SUPPORT_THRESHOLD` | 0.35 | 单句达标阈值 |

## 5. 失败模式与退化路径

| 故障 | 系统行为 |
| --- | --- |
| fastembed 模型未下载 / 无网络 | 回退哈希嵌入，全部测试仍绿 |
| faiss 未安装 | 回退 numpy 精确索引，与主实现交叉一致的测试自动跳过 |
| cross-encoder 不可用 | 回退余弦重排 |
| LLM 未配置密钥 | 默认 MockLLM，端到端可复现 |
| 检索无结果 | 空证据直接拒答（I-X4） |
| 生成内容无据 | 覆盖率不足拒答（I-X2） |

设计原则：**任何单点依赖缺失都只降低能力，不破坏正确性与可验证性。**
