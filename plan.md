# 项目: 基于 HotpotQA 的 RAG 双流水线评测项目

## 背景与目标

这个项目用于准备 AI 工程师面试,目标不是只做一个能回答问题的 RAG demo,而是构建一套**可量化、可消融、可复现实验结果的 RAG 系统评测流水线**。

新的评测集统一使用 **HotpotQA**。HotpotQA 天然包含多跳问题、supporting facts、答案和候选 Wikipedia 上下文,适合评估检索召回、跨文档推理、生成 groundedness 和引用溯源。

项目主线拆成两条核心 pipeline:

- **Ingestion pipeline**: 文档解析、切分、元数据治理、embedding/index 构建与更新。
- **Retrieval pipeline**: 查询处理、检索融合、rerank、上下文压缩与扩展。

Generation 和 Evaluation 作为系统闭环模块,用于验证两条 pipeline 的真实效果。

## 技术栈与约束

- 编排框架: Haystack 2.x
- 评测集: HotpotQA
  - 使用 Hugging Face `hotpotqa/hotpot_qa` 的 `fullwiki` 配置
  - 默认 split 为 `validation`
  - 每条样本至少保留: question、answer、context、supporting_facts、type、level
  - supporting facts 用于构造检索 relevance labels
- 嵌入模型: Haystack `OpenAIDocumentEmbedder` + DashScope OpenAI-compatible endpoint
  - baseline: `text-embedding-v4`
  - 默认维度: 1024
  - API key: `DASHSCOPE_API_KEY`
  - 默认 base URL: `https://dashscope.aliyuncs.com/compatible-mode/v1`
  - 配置中记录模型维度、语言支持、是否适合非对称检索
- 向量存储: Chroma,通过 Haystack `chroma-haystack` 集成作为本地 dense index
- 稀疏检索: BM25 或 sparse retriever,先保留 chunk JSONL 作为 BM25 输入
- 混合检索: dense + sparse,支持 RRF 或加权融合
- Reranker: cross-encoder 或 bge-reranker 系列,可配置启停
- 生成模型: Haystack `OpenAIGenerator` + DashScope OpenAI-compatible endpoint
  - baseline: `qwen-flash`
  - API key: `DASHSCOPE_API_KEY`
  - 默认 base URL: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- 生成质量评估: RAGAS 或等价评测模块,需要支持降级
- Python 3.10+,依赖写入 `pyproject.toml`

## 总体设计要求

- 所有可变参数放到配置中,消融实验靠配置驱动,不要改代码。
- Ingestion、Retrieval、Generation、Evaluation 互相解耦,每个模块都能独立运行。
- 检索评估必须能在没有 LLM/API key 的情况下运行。
- 每次实验输出一行结构化结果,写入 `results/` 下的 JSONL/CSV。
- 业务输出记录端到端延迟、token 成本、检索开销和模型配置。
- component 级运行耗时交给 Haystack tracing / OpenTelemetry / MLflow,避免在业务组件中手写计时代码。
- 明确区分:
  - 文档级召回: 是否找回 supporting document
  - 句子级召回: 是否找回 supporting fact 所在句子
  - 生成质量: 答案是否相关、是否忠实于上下文

## Pipeline 1: Ingestion

Ingestion pipeline 的目标是把原始数据或真实业务文档转成可检索、可更新、可评测的索引。

### 1. Parsing

- 支持 HotpotQA 数据加载:
  - 读取 question、answer、context、supporting_facts、type、level
  - 默认通过 `datasets.load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="validation[:N]")`
  - 将每个 context title + sentences 转成可索引文档
  - 将 supporting_facts 映射成 gold labels
- 支持真实文档解析接口:
  - PDF parsing
  - HTML parsing
  - 后续可扩展 Markdown、DOCX、纯文本
- 输出统一 Document schema:
  - `id`
  - `content`
  - `source`
  - `title`
  - `section`
  - `date`
  - `permissions`
  - `metadata`
  - `gold_label`

### 2. Chunking

- 实现基础 chunking:
  - 固定 token/字符窗口
  - overlap
  - 按标题、段落、句子边界切分
- 元数据继承:
  - 来源 source
  - 标题 title
  - 章节 section
  - 日期 date
  - 权限 permissions
  - HotpotQA supporting fact 标记
- 支持 contextual retrieval:
  - 参考 Anthropic Contextual Retrieval
  - 对每个 chunk 使用 whole parent document + chunk content 构造 prompt
  - 通过 Haystack `OpenAIGenerator` 调用轻量模型生成简短 chunk-specific context
  - 将生成的 contextual text prepend 到 chunk content 后再进入 BM25 JSONL 和 dense embedding
  - 用配置控制是否启用,方便做消融
- 需要保留 chunk 与父文档关系:
  - `chunk_id`
  - `parent_doc_id`
  - `prev_chunk_id`
  - `next_chunk_id`

### 3. Vector DB Indexing

- Dense indexing:
  - 支持 DashScope `text-embedding-v4` baseline
  - 通过 Haystack `OpenAIDocumentEmbedder` 接入 OpenAI-compatible embedding API
  - 在配置中记录:
    - provider
    - 模型名称
    - embedding 维度
    - API base URL
    - batch size
    - 语言支持
    - 是否推荐 query/document 非对称编码
    - normalize 策略
- Sparse / BM25 indexing:
  - 为同一批 chunk 建立 BM25 或 sparse index
  - 保留与 dense index 相同的 document ids
- 存储层:
  - 起步使用 ChromaDocumentStore
  - 本地持久化路径默认 `data/chroma_hotpotqa`
  - collection 默认 `hotpotqa_chunks`
  - 后续可替换为 Qdrant / Weaviate / Elasticsearch
  - 替换存储层不应影响 pipeline 上层逻辑
- BM25 / sparse indexing:
  - Chroma 主要承担 dense vector index
  - chunk JSONL 作为可审计的 chunk 输出产物
  - Haystack `InMemoryDocumentStore` 保存为 `data/hotpotqa_bm25_store.json`,用于 BM25 retrieval
  - Retrieval pipeline 中再将 Chroma dense 结果与 BM25 结果做 RRF 或加权融合

### 4. Embedding / Index 更新机制

- 支持全量重建:
  - 清空 index
  - 重新 parse、chunk、embed、write
- 支持增量更新:
  - 基于 source/document hash 判断内容变化
  - 新增文档: 写入新 chunk
  - 修改文档: 删除旧 chunk 后重新写入
  - 删除文档: 根据 parent_doc_id 删除相关 chunk
- 记录 index manifest:
  - 数据集版本
  - chunking 配置
  - embedding 模型
  - index 时间
  - document/chunk 数量

## Pipeline 2: Retrieval

Retrieval pipeline 的目标是从用户 query 出发,稳定找回可支持答案生成的上下文,并尽量减少无关 token。

当前实现要求: Retrieval 必须通过 Haystack `Pipeline` graph 编排,而不是在普通 Python 函数中手写阶段顺序。

### 1. Query Processing

- Query rewrite:
  - 改写口语化或省略 query
  - HotpotQA 中可测试原 query vs rewritten query
- Query expand:
  - 增加同义词、实体别名、相关概念
  - 记录 expand 后的 query variants
- HyDE:
  - 生成 hypothetical answer/document
  - 用 HyDE 文本参与 dense retrieval
  - 配置开关控制是否启用
- Query routing:
  - 判断走 dense、sparse、hybrid 或特定 metadata filter
  - 简单 baseline 可用规则
  - 后续可用轻量分类器或 LLM router
  - 当前实现: 配置驱动 route,支持 `hybrid` / `dense` / `bm25`

### 2. Metadata Filtering

- 支持按 metadata 过滤:
  - source
  - title
  - section
  - date
  - permissions
  - document type
- HotpotQA 实验中可模拟:
  - 限定 title 范围
  - 排除无权限文档
  - 按 level/type 分组评估

### 3. Hybrid Search

- Dense retrieval:
  - 向量相似度 top_k
- Sparse / BM25 retrieval:
  - 关键词 top_k
- 融合策略:
  - RRF: `score += 1 / (rrf_k + rank)`
  - 加权融合: dense 和 BM25 score 分别归一化后按权重加权
  - 可配置 dense_weight / sparse_weight
- 输出统一候选列表:
  - document/chunk id
  - dense score
  - sparse score
  - fused score
  - rank

### 4. Rerank

- 对 hybrid search 的候选结果做 cross-encoder rerank。
- 记录 rerank 前后指标变化:
  - Recall@k
  - MRR
  - nDCG
  - supporting fact hit rate
- reranker latency 通过 Haystack tracing 观测,不写入业务组件输出。
- 支持配置:
  - 是否启用
  - reranker 模型
  - rerank 输入 top_n
  - rerank 输出 top_k
  - 当前实现: 默认关闭 rerank,可选 `SentenceTransformersSimilarityRanker`

### 5. 上下文压缩 / 去重

- 目标: 把检索结果裁剪成更适合 LLM prompt 的上下文。
- 实现:
  - 删除重复 chunk
  - 合并来自同一 parent doc 的相邻 chunk
  - 裁掉与 query 低相关的句子
  - 限制总 token budget
  - 当前实现: chunk 去重 + 单文档字符预算裁剪
- 评估:
  - 压缩前后 token 数
  - gold supporting facts 是否仍被保留
  - 生成质量是否下降

### 6. Chunk 扩展

- 支持 small-to-big / parent-document retrieval:
  - 检索时命中小 chunk
  - 喂给 LLM 时取回相邻 chunk 或父文档片段
- 扩展策略:
  - prev/next chunk
  - parent section
  - parent document excerpt
  - 当前实现: 根据 `parent_doc_id` 从 BM25 store 找回同一 parent 的 chunks,合并为 parent context
- 记录:
  - 扩展前 chunk 数
  - 扩展后 context token 数
  - supporting fact 覆盖率

## Generation

Generation 模块用于把 query 和检索结果组织成答案,并检查答案是否可溯源。

当前实现要求: RAG 必须通过 Haystack `Pipeline` graph 接上 retrieval、answer generation 和 evaluation,CLI 只负责传入 pipeline inputs。

### 1. No-answer / 兜底

- 当检索分数过低或上下文不足时,允许输出 no-answer。
- 当生成后的 groundedness 低于阈值时,允许转换为 no-answer。
- 兜底策略:
  - 说明未找到足够依据
  - 返回最相关来源
  - 不编造答案

### 2. Prompt 拼装

- 将 query + 检索结果组织成 prompt。
- 每个 chunk 附带稳定 citation id。
- Prompt 中明确要求:
  - 只基于给定上下文回答
  - 无依据时返回 no-answer
  - 输出引用

### 3. 生成

- Haystack `OpenAIGenerator` 作为默认 generator,通过 OpenAI-compatible endpoint 接入 DashScope/Qwen。
- 模型名称、temperature、max tokens、prompt 模板全部配置化。
- 记录端到端生成 latency 和 token 数。

### 4. Citation / Attribution

- 答案中标注来源 chunk。
- 每个 citation 能回溯到:
  - chunk_id
  - parent_doc_id
  - title
  - source
  - supporting fact 标记

### 5. 幻觉检测 / Groundedness 校验

- 检查答案是否真的基于检索内容。
- 可选方法:
  - 当前实现: answer content words vs retrieved context 的轻量 lexical support
  - RAGAS faithfulness
  - answer sentence vs retrieved context entailment
  - citation coverage 检查
- 评估失败不能让整条流水线崩溃,需要记录错误并继续。

## Evaluation

Evaluation 是项目核心交付物,没有评估就无法调 chunking 阈值、embedding 模型、hybrid 权重、rerank top_n 或 prompt。

### 1. 端到端 / 系统级指标

- latency:
  - ingestion latency
  - retrieval latency p50/p95
  - generation latency
  - component latency 通过 tracing 后端分析
- cost:
  - 每 query token 数
  - embedding 开销
  - rerank 开销
  - generation 开销
- hit rate:
  - answer hit
  - supporting document hit
  - supporting fact hit

### 2. 检索质量

- Recall@k
- Precision@k
- MRR
- nDCG
- supporting document recall
- supporting fact recall
- 分组评估:
  - bridge
  - comparison
  - easy / medium / hard

### 3. 生成质量

- faithfulness / groundedness
- answer relevance
- exact match / F1,对齐 HotpotQA 标准答案
- citation precision / recall
- no-answer 触发率和误触发率

### 4. 评测框架

- 优先预留 RAGAS 接入点。
- RAGAS 不可用时,检索评估和基础 EM/F1 仍能独立运行。
- LLM-as-judge 后续通过配置接入,并提供:
  - retry
  - parse failure 记录
  - timeout
  - 小批量 smoke eval

## 分阶段实施计划

### 阶段 0: 重置项目骨架

- 删除旧 StratRAG 代码。
- 保留基础目录:
  - `data/`
  - `src/`
  - `tests/`
  - `results/`
- 更新 README 和依赖说明。
- 验收:
  - 仓库中不再保留 StratRAG loader/test/fixture
  - `plan.md` 明确使用 HotpotQA

### 阶段 1: HotpotQA 数据加载与 gold label 构造

- 加载 HotpotQA。
- 标准化样本:
  - question
  - answer
  - context
  - supporting_facts
  - type
  - level
- 将 context 转成 Document/chunk 初始结构。
- 根据 supporting_facts 构造 document-level 和 sentence-level relevance labels。
- 验收:
  - 打印数据集统计
  - 抽样展示 1 条完整样本
  - 输出 gold label 覆盖率

### 阶段 2: Ingestion pipeline baseline

- 实现 parsing -> chunking -> Chroma dense indexing -> chunk JSONL 输出。
- 建立 index manifest。
- 支持 DashScope `text-embedding-v4` baseline。
- 验收:
  - 成功写入 HotpotQA 小样本 Chroma index
  - 成功输出 chunk JSONL 供 BM25 使用
  - 输出 chunk 数、embedding 维度、端到端 index latency
  - 支持重复运行和全量重建

### 阶段 3: Retrieval pipeline baseline

- 实现 query -> dense/BM25/hybrid -> top_k。
- 实现 Recall@k、Precision@k、MRR、nDCG。
- 验收:
  - dense、BM25、hybrid 三组结果可对比
  - 指标写入 `results/`
  - 不依赖任何 LLM/API key

### 阶段 4: Rerank、上下文压缩与 chunk 扩展

- 加入 reranker。
- 加入去重、压缩、small-to-big 扩展。
- 对比启用前后的检索质量和 token 数。
- 验收:
  - 通过 tracing 观察 rerank latency
  - 输出压缩前后 token 变化
  - 输出 supporting fact 保留率

### 阶段 5: Generation 与 citation

- 接入 Haystack `PromptBuilder` + `OpenAIGenerator` generator。
- 实现 prompt 拼装、no-answer、citation。
- 验收:
  - 小批量问题能生成答案
  - 每个答案带 citation
  - no-answer 策略可配置

### 阶段 6: Evaluation 与消融实验

- 接入 RAGAS 或等价 groundedness 评估。
- 跑配置组合:
  - chunk size / overlap
  - contextual retrieval on/off
  - text-embedding-v4 / 后续备选 embedding 模型
  - dense / sparse / hybrid
  - RRF / weighted fusion
  - reranker on/off
  - compression on/off
  - small-to-big on/off
- 生成最终对比表:
  - Recall@k
  - Precision@k
  - MRR
  - nDCG
  - faithfulness
  - answer relevance
  - EM/F1
  - citation quality
  - p50/p95 latency
  - per-query cost
- 验收:
  - CSV/JSONL 结果表
  - 简单图表
  - README 中给出如何复现实验

## 编码规范

- 类型标注 + docstring。
- 关键步骤用 logging,不要用大量 print。
- 每个阶段有独立入口,例如:
  - `python -m src.stage1_hotpotqa`
  - `python -m src.stage2_ingestion`
  - `python -m src.stage3_retrieval`
- 每个 pipeline 至少有 smoke test。
- 不静默吞异常。
- 不把模型名、top_k、chunk size、prompt 模板写死在业务逻辑里。

## 当前交付边界

当前已经具备:

- HotpotQA ingestion baseline
- Chroma dense index
- Haystack BM25 store
- hybrid retrieval
- retrieval Haystack Pipeline graph
- rerank/压缩/parent expansion 开关
- generation prompt、生成、citation、no-answer
- retrieval/generation/system evaluation helper
- RAG Haystack Pipeline graph

下一步重点是批量评测 runner,把单 query 的 JSON 输出聚合为 JSONL/CSV,并接入 HotpotQA answer EM/F1 与可选 RAGAS。
