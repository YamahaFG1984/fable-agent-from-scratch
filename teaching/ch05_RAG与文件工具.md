# 第 5 章：RAG 与文件工具（教学笔记）

> 对应 notebook：`notebooks/ch05/ch05_rag_and_file_tools.ipynb`（Listing 5.1–5.31）
> 核心代码：`scratch_agents/rag.py`、`callbacks.py`、`tools/file_tools.py`
> 章节快照：`notebooks/ch05/agent.py`
>
> **TS 示例说明**：本文每段 Python 代码后附带一段功能对应的 TypeScript 代码，方便对照阅读。库映射约定：`client.embeddings.create`→`openai`（npm 包）的同名方法；`sklearn.metrics.pairwise.cosine_similarity` 在 TS/JS 生态没有直接平替，**手写 `cosineSimilarity(a, b)` 点积/模长函数**代替；`tiktoken` 精确计量 token → `js-tiktoken`（`encodingForModel`）；`pandas`（CSV/Excel → markdown 表格）在 TS 里没有一行到位的方法，CSV 用 `papaparse` 解析、Excel 用 `xlsx`（SheetJS）读取，再**手写循环拼出 markdown 表格字符串**；PyMuPDF 文本抽取 → `pdfjs-dist`，页面渲染成图像在 Node 环境较复杂，只用注释说明思路；图像/音频转 base64 用 Node 内置 `fs.readFileSync(...).toString("base64")`；回调机制（`before_tool_callbacks`/`after_tool_callbacks`）是本书自制框架 `scratch_agents` 的概念，没有官方 TS 移植，`ToolCall`/`ToolResult` 等类型沿用第 4 章教学笔记里建立的 TS interface 风格（`toolCallId`/`name`/`status`/`content` 同名对照）。

第 4 章的 Agent 能搜网页、能算数，但面对两类数据仍然瞎：**模型没见过的私有数据**（你的文件、内部文档）和**塞不进上下文的海量数据**（一次搜索返回几十万 token）。本章给出两把钥匙：**向量检索（RAG）** 和 **结构化探索（文件工具）**，最后用**回调机制**把它们优雅地挂进 Agent。

## 1. 问题与检索方法分类（5.1–5.2，概念节）

给 Agent 用内部数据，本质是"在正确的时机把正确的片段放进上下文"。检索方法四类：关键词检索、**向量检索**（按语义相似度）、图检索、**结构化检索**（按目录/文件结构导航）。本章实现后两者中的向量与结构化两种 —— 它们分别对应"数据无结构"和"数据有结构"两种场景。

## 2. 向量检索三件套（5.3，Listing 5.1–5.6）

### ① Embedding：文本 → 向量（Listing 5.1–5.2）

```python
def get_embeddings(texts, model="text-embedding-3-small"):
    if isinstance(texts, str):
        texts = [texts]
    response = client.embeddings.create(input=texts, model=model)
    return np.array([item.embedding for item in response.data])
```

```typescript
async function getEmbeddings(
  texts: string | string[],
  model = "text-embedding-3-small"
): Promise<number[][]> {
  if (typeof texts === "string") {
    texts = [texts];
  }
  const response = await client.embeddings.create({ input: texts, model });
  // TS 没有 numpy，这里直接返回 number[][]（二维数组）即可，后续向量运算手写实现
  return response.data.map((item) => item.embedding);
}
```

语义相近 → 向量夹角小。经典演示：

```
"The cat is sleeping..." vs "A kitten is playing..."  → 相似度高（cat≈kitten）
"The cat is sleeping..." vs "The dog is running..."   → 相似度低
```

**关键认知：embedding 捕捉的是语义而不是字面**。"cat" 和 "kitten" 没有一个共同字母的词根，但向量空间里是邻居。度量用余弦相似度（`sklearn.cosine_similarity`）。

### ② Chunking：长文切块（Listing 5.3–5.4）

```python
def fixed_length_chunking(text, chunk_size=500, overlap=50):
    while start < len(text):
        chunk = text[start:start+chunk_size].strip()
        ...
        start = end - overlap if end < len(text) else end   # 相邻块重叠 50 字符
```

```typescript
function fixedLengthChunking(text: string, chunkSize = 500, overlap = 50): string[] {
  const chunks: string[] = [];
  let start = 0;

  while (start < text.length) {
    const end = start + chunkSize;
    const chunk = text.slice(start, end).trim();
    if (chunk) {
      chunks.push(chunk);
    }
    start = end < text.length ? end - overlap : end;   // 相邻块重叠 50 字符
  }

  return chunks;
}
```

为什么切块？embedding 对整篇长文只能给一个"平均语义"，检索粒度太粗。为什么 **overlap**？防止一句话正好被切断在边界上，两个块各拿半句都检索不到。fixed-length 是最笨但最稳的策略（还有按句/按段/语义切分等进阶方案）。

### ③ Vector Search：top-k 检索（Listing 5.5–5.6）

```python
def vector_search(query, chunks, chunk_embeddings, top_k=3):
    query_embedding = get_embeddings(query)
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_k]   # 相似度降序取前 k
    return [{'chunk': chunks[i], 'similarity': similarities[i]} for i in top_indices]
```

```typescript
// TS/JS 生态没有 sklearn.metrics.pairwise.cosine_similarity 的直接平替，手写点积/模长实现
function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function vectorSearch(
  query: string,
  chunks: string[],
  chunkEmbeddings: number[][],
  topK = 3
): Promise<{ chunk: string; similarity: number }[]> {
  const [queryEmbedding] = await getEmbeddings(query);
  const similarities = chunkEmbeddings.map((emb) => cosineSimilarity(queryEmbedding, emb));

  const topIndices = similarities
    .map((sim, i) => [sim, i] as const)
    .sort((a, b) => b[0] - a[0])          // 相似度降序取前 k
    .slice(0, topK)
    .map(([, i]) => i);

  return topIndices.map((i) => ({ chunk: chunks[i], similarity: similarities[i] }));
}
```

测试很有说服力：查询 "Artificial Intelligence" 在 4 个文档里排出 machine learning / deep learning 在前、"Cats are popular pets" 垫底 —— 查询词一个都没出现在文档里，纯靠语义。

## 3. 实战练习：RAG 用作"上下文压缩器"（5.3.4，Listing 5.7–5.11）⭐

这是本章对 RAG 最独特的定位。传统教程把 RAG 讲成"企业知识库问答"，本书的第一个应用场景却是：**压缩工具输出**。

流程（数字会因搜索结果而异）：

1. Tavily 搜索带 `include_raw_content=True` → 返回 10 篇网页全文
2. `tiktoken` 数一下：**几十万字符、十几万 token** —— 直接塞给模型既贵又可能爆上下文
3. 全部切块 + embedding
4. 用真正关心的问题（如 "quantum computing"）向量检索 top-3
5. 再数 token：**节省 90%+**

> RAG 的本质不是"知识库"，而是**信息过滤**：从海量候选文本里挑出与当前问题语义最相关的一小撮。知识库问答只是它的应用之一，压缩搜索结果、压缩文件内容同样是。

这个练习在 5.5 节会被包装成正式的回调组件。

## 4. 结构化检索：文件系统工具（5.4，Listing 5.12–5.20）

场景：GAIA 有一批带附件（zip/xlsx/png/mp3/pdf）的题，纯文本 Agent 零分。数据有天然结构（目录树），用导航式探索比向量检索更合适。

先准备可复现实验环境（Listing 5.13）：`reset_workspace()` 每次把缓存的 GAIA 附件重新拷贝到工作区 —— **Agent 会真实地改动文件系统，可复现性要求每次实验从干净状态开始。**

四个工具，每个都体现"为 LLM 消费而设计输出"：

**`unzip_file`**（Listing 5.15）：解压后返回文件清单摘要（最多列 20 个）—— 不是返回 `None`，而是告诉模型"你现在有什么可以探索了"。

**`list_files`**（Listing 5.16）：目录在前文件在后、跳过隐藏文件、目录带 `/` 后缀 —— 输出格式为模型的"下一步决策"优化。

**`read_file`**（Listing 5.17–5.19）：按扩展名分派的多格式阅读器：

```python
if ext in TEXT_EXTENSIONS:   return _read_text_file(...)   # 带行号，支持行范围
elif ext == '.csv':          return _read_csv(...)          # pandas → markdown 表格
elif ext in SPREADSHEET_EXTENSIONS: return _read_excel(...) # 同上
```

```typescript
if (TEXT_EXTENSIONS.includes(ext)) {
  return readTextFile(filePath, startLine, endLine);   // 带行号，支持行范围
} else if (ext === ".csv") {
  return readCsv(filePath);       // 手写实现 → markdown 表格，见下
} else if (SPREADSHEET_EXTENSIONS.includes(ext)) {
  return readExcel(filePath);     // 同上
}

// pandas 的 df.to_markdown() 在 TS 里没有一行到位的等价物，
// 这里用 papaparse / xlsx 解析数据后手写循环拼出 markdown 表格字符串
import Papa from "papaparse";
import * as XLSX from "xlsx";
import { readFileSync } from "node:fs";

function rowsToMarkdown(header: string[], rows: string[][]): string {
  const headerLine = `| ${header.join(" | ")} |`;
  const sepLine = `| ${header.map(() => "---").join(" | ")} |`;
  const bodyLines = rows.map((r) => `| ${r.join(" | ")} |`);
  return [headerLine, sepLine, ...bodyLines].join("\n");
}

function readCsv(filePath: string): string {
  const content = readFileSync(filePath, "utf-8");
  const { data } = Papa.parse<string[]>(content, { skipEmptyLines: true });
  const [header, ...rows] = data;
  return rowsToMarkdown(header, rows);
}

function readExcel(filePath: string): string {
  const workbook = XLSX.readFile(filePath);
  const sheet = workbook.Sheets[workbook.SheetNames[0]];
  const data = XLSX.utils.sheet_to_json<string[]>(sheet, { header: 1 });
  const [header, ...rows] = data as string[][];
  return rowsToMarkdown(header, rows);
}
```

两个细节：文本带**行号**（`{i:4d} | line`，模型可以说"第 42 行有问题"）；表格转 **markdown**（LLM 对 markdown 表的理解远好于原始 CSV 逗号流）。

**`read_media_file`**（Listing 5.20）：图像/音频/PDF 的统一入口 —— **工具内部再调一次多模态 LLM**：

- 图像 → base64 → `gpt-5.5` 视觉问答
- 音频 → base64 → `gpt-audio`
- PDF → PyMuPDF 抽文本（前 3000 字符）+ 前 5 页渲染成图 → 一起发给视觉模型（文本抽取会丢排版/图表，页面图像补上这部分信息）

签名是 `read_media_file(file_path, query)` —— 主 Agent 不拿原始像素，而是**委托一个子模型去看，只拿回答案文本**。这是"LLM 在工具里"的委托模式，第 9 章 Agent-as-Tool 是它的推广。

### 组装与实测（Listing 5.21–5.23）

```python
tools = [search_web, tool(unzip_file), tool(list_files), tool(read_file), tool(read_media_file)]
agent = Agent(model=LlmClient(model="gpt-5.5"), tools=tools, max_steps=20)
prompt = f"{problem['Question']}\n\nThe attached file is located at: {file_path}"
```

```typescript
const tools = [searchWeb, tool(unzipFile), tool(listFiles), tool(readFile), tool(readMediaFile)];
const agent = new Agent({ model: new LlmClient({ model: "gpt-5.5" }), tools, maxSteps: 20 });
const prompt = `${problem.Question}\n\nThe attached file is located at: ${filePath}`;
```

注意两点：`max_steps` 提到 20（探索类任务步数多）；文件路径由**代码注入提示词**，不让模型猜（又是第 3 章的准则）。Agent 自主完成 unzip → list → read → 综合作答的全链条，没有任何硬编码流程。

## 5. 回调机制：不改内核的扩展点（5.5，Listing 5.24–5.31）⭐

需求出现了：想在工具执行前审批、执行后压缩结果。改 `act()` 硬编码？那每加一个需求都要动内核。本章的答案是**中间件模式**：

```python
Agent(..., before_tool_callbacks=[...], after_tool_callbacks=[...])
```

```typescript
new Agent({ /* ... */, beforeToolCallbacks: [/* ... */], afterToolCallbacks: [/* ... */] });
```

`act()` 重构成三段（Listing 5.25）：

```
Stage 1  before 回调链:  callback(context, tool_call)
         返回 None → 放行；返回值 → 该值直接成为工具结果，跳过真实执行
Stage 2  真实执行（仅当没被拦截）
Stage 3  after 回调链:   callback(context, tool_result)
         返回 None → 原样；返回 ToolResult → 替换
```

**协议就一条：`None` = 不干预，非 None = 接管。** 回调同时支持同步/异步（`inspect.isawaitable`）。

### 应用一：人工审批（before，Listing 5.26–5.28）

```python
DANGEROUS_TOOLS = ["delete_file", "send_email", "execute_sql"]

def approval_callback(context, tool_call):
    if tool_call.name not in DANGEROUS_TOOLS:
        return None                                  # 安全工具直接放行
    response = input("Do you want to execute? (y/n): ")
    if response == 'y':
        return None                                  # 批准 → 放行
    return f"User denied execution of {tool_call.name}"  # 拒绝 → 这句话成为工具结果
```

```typescript
const DANGEROUS_TOOLS = ["delete_file", "send_email", "execute_sql"];

async function approvalCallback(
  context: ExecutionContext,
  toolCall: ToolCall
): Promise<string | null> {
  if (!DANGEROUS_TOOLS.includes(toolCall.name)) {
    return null;                                     // 安全工具直接放行
  }
  // Node 没有 Python input() 那样的同步阻塞读取，用 readline/promises 做等价的异步提示
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  const response = await rl.question("Do you want to execute? (y/n): ");
  rl.close();

  if (response === "y") {
    return null;                                     // 批准 → 放行
  }
  return `User denied execution of ${toolCall.name}`; // 拒绝 → 这句话成为工具结果
}
```

精妙处在拒绝分支：**拒绝信息作为工具结果进入历史**，模型看到"用户拒绝了"，会礼貌地换方案而不是崩溃。局限也很明显：`input()` 是同步阻塞的，Web 服务里不可用 —— 第 6 章会用"暂停-恢复"机制（`pending_confirmation`）彻底重做这个功能。

### 应用二：搜索压缩（after，Listing 5.29–5.31）

把第 3 节的练习组件化：

```python
def search_compressor(context, tool_result):
    if tool_result.name != "search_web": return None
    if len(tool_result.content[0]) < 2000: return None      # 短结果不折腾
    query = _extract_search_query(context, tool_result.tool_call_id)  # 从历史反查原始查询
    chunks = fixed_length_chunking(original_content, 500, 50)
    results = vector_search(query, chunks, get_embeddings(chunks), top_k=3)
    return ToolResult(..., content=["\n\n".join(r['chunk'] for r in results)])
```

```typescript
async function searchCompressor(
  context: ExecutionContext,
  toolResult: ToolResult
): Promise<ToolResult | null> {
  if (toolResult.name !== "search_web") return null;
  const originalContent = toolResult.content[0] as string;
  if (originalContent.length < 2000) return null;              // 短结果不折腾

  const query = extractSearchQuery(context, toolResult.toolCallId);  // 从历史反查原始查询
  if (!query) return null;

  const chunks = fixedLengthChunking(originalContent, 500, 50);
  const results = await vectorSearch(query, chunks, await getEmbeddings(chunks), 3);

  return {
    ...toolResult,
    content: [results.map((r) => r.chunk).join("\n\n")],
  };
}
```

细节 `_extract_search_query`（Listing 5.30）：压缩需要知道"用户搜了什么"才能算相关性，答案在 `context.events` 里 —— 用 `tool_call_id` 反查对应的 ToolCall 拿到 query 参数。**这就是第 4 章把一切都记进 events 的回报：任何组件都能回溯完整历史。**

最终形态：一个 Agent 同时挂两个回调，审批管安全，压缩管成本，互不知晓、随意插拔。

---

## 本章要带走的 5 句话

1. RAG 三件套：**embedding（语义→向量）→ chunking（固定长度+overlap）→ 余弦 top-k 检索**；它的本质是信息过滤，不只是知识库。
2. 本书 RAG 的第一个用途是**压缩工具输出**（几十万 token → 节省 90%+），知识库问答反而是次要场景。
3. 文件工具的设计准则：输出为 LLM 消费而格式化（行号、markdown 表、解压清单），路径等系统信息由代码注入提示词。
4. `read_media_file` 展示了"工具里再调一个 LLM"的委托模式 —— 主 Agent 不看像素，只拿子模型的文字结论。
5. 回调 = Agent 的中间件：**before 可拦截（审批），after 可改写（压缩），`None` 即放行** —— 不改内核就能加横切能力，而"拒绝也进历史"让模型能优雅应对被拒。
