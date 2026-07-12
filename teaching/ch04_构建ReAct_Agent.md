# 第 4 章：构建 ReAct Agent（教学笔记）

> 对应 notebook：`notebooks/ch04/ch04_react_agent.ipynb`（Listing 4.1–4.34）
> 核心代码：`scratch_agents/types.py`、`context.py`、`llm.py`、`agent.py`、`tools/base.py`
> 章节快照：`notebooks/ch04/{base,context,agent}.py`（该章时刻的精简版，不含后面章节的功能）
>
> **TS 示例说明**：本文每段 Python 代码后附带一段功能对应的 TypeScript 代码，方便对照阅读。映射约定：`pydantic.BaseModel` → 纯数据容器用 `interface`，需要运行时校验/作为 LLM 输出 schema 的用 `zod` 的 `z.object`；`Literal["a","b","c"]` → 联合字符串字面量类型 `"a" | "b" | "c"`；`@dataclass` → 普通 TS `class`（构造函数赋值字段）；`ABC` + `@abstractmethod` → TS `abstract class` + `abstract` 方法；`Union[...]` → TS 联合类型 `A | B | C`；`async def`/`await` → TS `async function`/`await`；`inspect.signature` 之类的运行时反射在 TS 中不存在，需要显式声明或手写 JSON schema/zod schema 代替。`scratch_agents` 是本书随 Python 代码给出的自制教学框架，没有官方 TS 移植，本章的 TS 示例都是**结构对照式翻译**（保持相同的类结构、字段、方法签名和控制流），帮助读者对照理解，不是可直接运行的库。

第 3 章结尾的 `simple_agent_loop` 只有 20 行，但它有致命缺陷：会死循环、状态和逻辑搅在一起、换厂商要改核心代码、没法扩展。本章把它重构成一个真正的框架 —— **全书后面 6 章都只是往这个骨架上挂功能，骨架本身不再变。**

## 1. ReAct：思考-行动循环（4.1）

ReAct = **Rea**soning + **Act**ing。Agent 的本质是一个循环：

```
思考（LLM 决定下一步）→ 行动（执行工具）→ 观察（结果进入历史）→ 再思考 → … → 最终回答
```

历史注解：原始 ReAct 论文（2022）靠提示词让模型输出 `Thought: ... / Action: ...` 文本再用正则解析 —— 脆弱且易错。现代做法直接用第 3 章学的**原生 tool calling**，模型输出结构化的 `tool_calls`，解析问题消失了。本书用的就是现代做法。

## 2. 先看终点：我们要造出什么（Listing 4.1）

```python
from scratch_agents import Agent, LlmClient
from scratch_agents.tools import calculator, search_web

agent = Agent(
    model=LlmClient(model="gpt-5.4-mini"),
    tools=[calculator, search_web],
    instructions="You are a helpful assistant",
)
result = await agent.run("What is 1234 * 5678?")
```

```typescript
import { Agent, LlmClient } from "./scratchAgents";
import { calculator, searchWeb } from "./scratchAgents/tools";

const agent = new Agent({
  model: new LlmClient({ model: "gpt-5.4-mini" }),
  tools: [calculator, searchWeb],
  instructions: "You are a helpful assistant",
});
const result = await agent.run("What is 1234 * 5678?");
```

用户体验三行搞定。为了支撑这三行，需要四个组件，职责严格分离：

| 组件 | 职责 | 一句话 |
|------|------|--------|
| `ExecutionContext` | 状态存储 | 发生过的一切 |
| `BaseTool` / `FunctionTool` | 能力抽象 | Agent 能做什么 |
| `LlmClient` + `LlmRequest/Response` | 通信层 | 和模型怎么说话 |
| `Agent` | 编排 | 循环本身 |

**核心设计思想（4.2.2 信息流）**：`context.events` 是唯一事实来源（single source of truth）；每次调 LLM 前，从 events **现做**一个 `LlmRequest` 视图发出去。存储和呈现分离 —— 这一刀切下去，第 6 章的上下文压缩才有了施展空间（改视图，不动历史）。

## 3. ExecutionContext：Agent 的中央存储（4.3，Listing 4.2–4.4）

### 三种内容原子（ContentItem）

```python
class Message(BaseModel):      # 文本消息
    role: Literal["system", "user", "assistant"]
    content: str

class ToolCall(BaseModel):     # LLM 的调用请求
    tool_call_id: str
    name: str
    arguments: dict            # 已解析成 dict（不再是 JSON 字符串）

class ToolResult(BaseModel):   # 执行结果
    tool_call_id: str
    name: str
    status: Literal["success", "error"]   # ← 错误是一等公民
    content: list

ContentItem = Union[Message, ToolCall, ToolResult]
```

```typescript
interface Message {              // 文本消息
  role: "system" | "user" | "assistant";
  content: string;
}

interface ToolCall {             // LLM 的调用请求
  toolCallId: string;
  name: string;
  arguments: Record<string, unknown>;   // 已解析成对象（不再是 JSON 字符串）
}

interface ToolResult {           // 执行结果
  toolCallId: string;
  name: string;
  status: "success" | "error";   // ← 错误是一等公民
  content: unknown[];
}

type ContentItem = Message | ToolCall | ToolResult;
```

对比第 3 章直接用 OpenAI 的 dict：这三个类是**厂商中立的内部语言**。整个框架内部只说这套语言，OpenAI 格式被隔离到通信层的边界上。

### Event：带元数据的记录单元

```python
class Event(BaseModel):
    id: str                    # uuid
    execution_id: str          # 属于哪次执行
    timestamp: float
    author: str                # "user" 或 agent 名字 ← 多智能体的伏笔
    content: List[ContentItem]
```

```typescript
interface Event {
  id: string;                  // uuid
  executionId: string;         // 属于哪次执行
  timestamp: number;
  author: string;              // "user" 或 agent 名字 ← 多智能体的伏笔
  content: ContentItem[];
}
```

`author` 字段现在看着多余，到第 9 章多个 agent 共享一个 context 时就是身份标识。

### ExecutionContext 本体

```python
@dataclass
class ExecutionContext:
    execution_id: str
    events: List[Event]                 # 完整历史
    current_step: int = 0               # 防死循环的计数器
    state: Dict[str, Any]               # 任意共享状态（后面章节大量使用）
    final_result: Optional[str | BaseModel] = None   # 一旦非 None，循环结束
```

```typescript
class ExecutionContext {
  executionId: string;
  events: Event[] = [];                       // 完整历史
  currentStep: number = 0;                    // 防死循环的计数器
  state: Record<string, unknown> = {};        // 任意共享状态（后面章节大量使用）
  finalResult: string | Record<string, unknown> | null = null;   // 一旦非 null，循环结束

  constructor(executionId: string) {
    this.executionId = executionId;
  }
}
```

它就是第 3 章那个 `messages` 列表的结构化升级版，外加：步数计数、自由状态区、结束标志。

## 4. 工具抽象（4.4，Listing 4.5–4.8）

### BaseTool：统一接口

```python
class BaseTool(ABC):
    def __init__(self, name=None, description=None, tool_definition=None): ...

    @property
    def tool_definition(self): return self._tool_definition   # 给 LLM 看的 schema

    @abstractmethod
    async def execute(self, context: ExecutionContext, **kwargs): ...

    async def __call__(self, context, **kwargs):
        return await self.execute(context, **kwargs)
```

```typescript
abstract class BaseTool {
  protected _toolDefinition?: Record<string, unknown>;

  constructor(protected name?: string, protected description?: string, toolDefinition?: Record<string, unknown>) {
    this._toolDefinition = toolDefinition;
  }

  get toolDefinition(): Record<string, unknown> | undefined {
    return this._toolDefinition;   // 给 LLM 看的 schema
  }

  abstract execute(context: ExecutionContext, kwargs: Record<string, unknown>): Promise<unknown>;

  // TS 没有 __call__ 魔术方法，用普通方法模拟"工具可直接被调用"
  async call(context: ExecutionContext, kwargs: Record<string, unknown>): Promise<unknown> {
    return this.execute(context, kwargs);
  }
}
```

关键决策：**`execute` 第一个参数永远是 context**。工具因此能读写执行状态（第 8 章 E2B 沙箱、第 9 章 transfer 全靠这个），但 context 不出现在 tool_definition 里 —— 模型不知道它的存在。这正是第 3 章"系统性数据由代码注入，不让模型提供"准则的落地。

### FunctionTool：普通函数 → 工具

```python
class FunctionTool(BaseTool):
    def __init__(self, func, ...):
        self.func = func
        self.needs_context = 'context' in inspect.signature(func).parameters
        # 名字 ← func.__name__，描述 ← docstring，schema ← 类型注解（第 3 章的内省）

    async def execute(self, context, **kwargs):
        result = self.func(context=context, **kwargs) if self.needs_context \
                 else self.func(**kwargs)
        if inspect.iscoroutine(result):   # 同步/异步函数通吃
            return await result
        return result
```

```typescript
class FunctionTool extends BaseTool {
  private func: (args: Record<string, unknown>) => unknown | Promise<unknown>;
  private needsContext: boolean;

  constructor(
    func: (args: Record<string, unknown>) => unknown | Promise<unknown>,
    needsContext: boolean,
    meta?: { name?: string; description?: string; toolDefinition?: Record<string, unknown> }
  ) {
    super(meta?.name, meta?.description, meta?.toolDefinition);
    this.func = func;
    // TS 缺乏 inspect.signature 那样的运行时反射，needsContext 需要在构造时显式传入
    // 名字 ← func.name，描述 ← 需手动传入或写 JSDoc，schema ← 需手写 zod/JSON schema（无第 3 章那样的类型注解内省）
    this.needsContext = needsContext;
  }

  async execute(context: ExecutionContext, kwargs: Record<string, unknown>): Promise<unknown> {
    const result = this.needsContext
      ? this.func({ context, ...kwargs })
      : this.func(kwargs);
    // JS 的 async 函数天然返回 Promise，用 instanceof Promise 判断而非区分"协程"
    if (result instanceof Promise) {   // 同步/异步函数通吃
      return await result;
    }
    return result;
  }
}
```

三个巧思：自动检测函数要不要 context；自动兼容 sync/async；定义生成复用第 3 章的 `function_to_input_schema`。配套的 `@tool` 装饰器让任何函数一行变工具。

### MCP 工具接入（Listing 4.7–4.8）

```python
def _create_mcp_tool(mcp_tool, connection) -> FunctionTool:
    async def call_mcp(**kwargs):                    # 闭包捕获连接信息
        async with stdio_client(...) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(mcp_tool.name, kwargs)
                return _extract_text_content(result)
    return FunctionTool(func=call_mcp, tool_definition={...  # 直接用 MCP 的 inputSchema
    })
```

```typescript
function createMcpTool(mcpTool: McpToolDef, connection: McpConnection): FunctionTool {
  async function callMcp(kwargs: Record<string, unknown>): Promise<unknown> {
    // 闭包捕获连接信息
    const { read, write } = await stdioClient(/* ... */);
    const session = new ClientSession(read, write);
    await session.initialize();
    const result = await session.callTool(mcpTool.name, kwargs);
    return extractTextContent(result);
  }
  return new FunctionTool(callMcp, /* needsContext */ false, {
    toolDefinition: mcpTool.inputSchema,   // 直接用 MCP 的 inputSchema
  });
}
```

`load_mcp_tools()` 连一次服务器、发现所有工具、每个包成 FunctionTool。抽象的威力：**MCP 工具进了框架之后和本地函数毫无区别**，Agent 根本不知道也不需要知道。（简化的代价：每次调用重建连接。）

## 5. LLM 通信层（4.5，Listing 4.9–4.14）

```python
class LlmRequest(BaseModel):        # 发什么
    instructions: List[str]         # → system 消息
    contents: List[ContentItem]     # → 对话历史
    tools: List[BaseTool]
    tool_choice: Optional[str]

class LlmResponse(BaseModel):       # 收什么
    content: List[ContentItem]      # 统一还是内部语言
    error_message: Optional[str]
    usage_metadata: Dict            # token 统计
```

```typescript
interface LlmRequest {              // 发什么
  instructions: string[];           // → system 消息
  contents: ContentItem[];          // → 对话历史
  tools: BaseTool[];
  toolChoice?: string;
}

interface LlmResponse {             // 收什么
  content: ContentItem[];           // 统一还是内部语言
  errorMessage?: string;
  usageMetadata: Record<string, unknown>;   // token 统计
}
```

`LlmClient.generate()` 做三件事：**翻译出去**（`_build_messages`）→ 调 LiteLLM → **翻译回来**（`_parse_response`）。

`_build_messages` 里有个值得注意的细节（Listing 4.12）：内部的 `ToolCall` 要合并回前一条 assistant 消息的 `tool_calls` 数组里 —— 因为 OpenAI 协议要求 assistant 消息与其工具调用是一条消息，而我们内部把它们拆成了独立原子。翻译层就是处理这种阻抗失配的地方。

错误处理哲学延续第 3 章：`generate()` 里 `except Exception → LlmResponse(error_message=...)`，**永不抛异常**，让上层统一处理。

## 6. Agent 本体：循环的工业化（4.6，Listing 4.15–4.23）

### run()：外循环

```python
async def run(self, user_input, context=None) -> AgentResult:
    if context is None:
        context = ExecutionContext()          # 可传入已有 context（多轮/多agent 的接口）

    context.add_event(Event(author="user",
                            content=[Message(role="user", content=user_input)]))

    while not context.final_result and context.current_step < self.max_steps:
        await self.step(context)
        if self._is_final_response(context.events[-1]):
            context.final_result = self._extract_final_result(context.events[-1])

    return AgentResult(output=context.final_result, context=context)
```

```typescript
async run(userInput: string, context?: ExecutionContext): Promise<AgentResult> {
  if (context === undefined) {
    context = new ExecutionContext(crypto.randomUUID());   // 可传入已有 context（多轮/多agent 的接口）
  }

  context.addEvent({
    author: "user",
    content: [{ role: "user", content: userInput } as Message],
  } as Event);

  while (context.finalResult === null && context.currentStep < this.maxSteps) {
    await this.step(context);
    const lastEvent = context.events[context.events.length - 1];
    if (this.isFinalResponse(lastEvent)) {
      context.finalResult = this.extractFinalResult(lastEvent);
    }
  }

  return { output: context.finalResult, context };
}
```

对比第 3 章 `while True` 的两处进化：**`max_steps` 保险丝**（模型抽风也最多烧 N 步）；**结果连同完整 context 一起返回**（可审计、可调试、可接着跑）。

结束判据和第 3 章完全一样，只是提炼成了方法：

```python
def _is_final_response(self, event) -> bool:
    # 既没有 ToolCall 也没有 ToolResult = 纯文本回答 = 结束
    return not has_tool_calls and not has_tool_results
```

```typescript
isFinalResponse(event: Event): boolean {
  // 既没有 ToolCall 也没有 ToolResult = 纯文本回答 = 结束
  return !hasToolCalls && !hasToolResults;
}
```

### step()：一次思考-行动周期

```python
async def step(self, context):
    llm_request = self._prepare_llm_request(context)   # 从 events 现做视图
    llm_response = await self.think(llm_request)       # 思考
    context.add_event(Event(author=self.name, content=llm_response.content))

    tool_calls = [c for c in llm_response.content if isinstance(c, ToolCall)]
    if tool_calls:
        tool_results = await self.act(context, tool_calls)   # 行动
        context.add_event(Event(author=self.name, content=tool_results))

    context.increment_step()
```

```typescript
async step(context: ExecutionContext): Promise<void> {
  const llmRequest = this.prepareLlmRequest(context);   // 从 events 现做视图
  const llmResponse = await this.think(llmRequest);     // 思考
  context.addEvent({ author: this.name, content: llmResponse.content } as Event);

  // TS 没有 isinstance，用字段特征判断代替 Python 的类型检查
  const toolCalls = llmResponse.content.filter(
    (c): c is ToolCall => "toolCallId" in c && "arguments" in c
  );
  if (toolCalls.length > 0) {
    const toolResults = await this.act(context, toolCalls);   // 行动
    context.addEvent({ author: this.name, content: toolResults } as Event);
  }

  context.incrementStep();
}
```

`think()` 只有一行（`return await self.model.generate(llm_request)`）—— 存在的意义是**语义和可覆写性**（子类可以在思考前后插逻辑）。

### act()：执行工具，错误进历史

```python
async def act(self, context, tool_calls) -> List[ToolResult]:
    for tool_call in tool_calls:
        if tool_call.name not in tools_dict:      # 模型编了个不存在的工具名？
            results.append(ToolResult(status="error",
                          content=[f"Tool '{tool_call.name}' not found"]))
            continue
        try:
            output = await tool(context, **tool_call.arguments)
            results.append(ToolResult(status="success", content=[output]))
        except Exception as e:
            results.append(ToolResult(status="error", content=[str(e)]))
    return results
```

```typescript
async act(context: ExecutionContext, toolCalls: ToolCall[]): Promise<ToolResult[]> {
  const results: ToolResult[] = [];
  for (const toolCall of toolCalls) {
    const tool = toolsDict[toolCall.name];
    if (tool === undefined) {   // 模型编了个不存在的工具名？
      results.push({
        toolCallId: toolCall.toolCallId,
        name: toolCall.name,
        status: "error",
        content: [`Tool '${toolCall.name}' not found`],
      });
      continue;
    }
    try {
      const output = await tool.call(context, toolCall.arguments);
      results.push({ toolCallId: toolCall.toolCallId, name: toolCall.name, status: "success", content: [output] });
    } catch (e) {
      results.push({ toolCallId: toolCall.toolCallId, name: toolCall.name, status: "error", content: [String(e)] });
    }
  }
  return results;
}
```

**任何失败都变成 `status="error"` 的 ToolResult 回到历史里** —— 模型下一步能看到自己犯了什么错并改正。Agent 的鲁棒性不是靠防止错误，而是靠把错误变成模型可见的信息。

## 7. 结构化输出：把工具当输出格式化器（4.7，Listing 4.24–4.29）⭐

需求：让 Agent 返回 Pydantic 对象而不是散文。第 2 章的 `response_format` 在这不好使 —— Agent 中间步骤要自由调用工具，不能全程强制 JSON。

**本章最聪明的技巧：最终答案本身也是一个工具。**

```python
# output_type 存在时，自动注入一个 final_answer 工具
@tool(name="final_answer", description="Return the final structured answer...")
def final_answer(output: self.output_type) -> self.output_type:
    return output
```

```typescript
// TS 没有函数装饰器语法糖，这里用一个返回 FunctionTool 的工厂函数模拟 @tool
function makeFinalAnswerTool<T>(outputType: z.ZodType<T>): FunctionTool {
  async function finalAnswer(args: { output: T }): Promise<T> {
    return args.output;
  }
  return new FunctionTool(finalAnswer as (args: Record<string, unknown>) => unknown, false, {
    name: "final_answer",
    description: "Return the final structured answer...",
    toolDefinition: { parameters: outputType },   // schema 直接来自 zod，而非运行时类型内省
  });
}
```

配合三处修改：

1. `tool_choice = "required"` —— 强制模型每步都必须调用工具，**堵死自由文本出口**，唯一的结束方式就是调 `final_answer`；
2. `_is_final_response`：改为检测 "`final_answer` 工具成功执行"；
3. `_extract_final_result`：从该 ToolResult 里取出 Pydantic 对象。

```python
class SentimentAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    key_phrases: List[str]

agent = Agent(model=..., output_type=SentimentAnalysis, ...)
result = await agent.run("This product exceeded my expectations!")
result.output.sentiment    # "positive" —— 是对象属性，不是要解析的文本
```

```typescript
// 这里的 output_type 要作为 LLM 输出 schema 使用，需要运行时校验，所以用 zod 而不是 interface
const SentimentAnalysis = z.object({
  sentiment: z.enum(["positive", "negative", "neutral"]),
  confidence: z.number(),
  keyPhrases: z.array(z.string()),
});
type SentimentAnalysis = z.infer<typeof SentimentAnalysis>;

const agent = new Agent({ model /* ... */, outputType: SentimentAnalysis /* ... */ });
const result = await agent.run("This product exceeded my expectations!");
result.output.sentiment;   // "positive" —— 是对象属性，不是要解析的文本
```

工具调用机制被复用成了输出约束机制 —— schema 校验、参数解析全都白送。

## 8. GAIA 实测：从 LLM 到 Agent（4.8，Listing 4.30–4.34）

第 2 章测过裸 LLM，现在同一套题、同一评分逻辑，换上 Agent（带 Tavily MCP 搜索工具 + `output_type=GaiaOutput` + `max_steps=15`）：

```python
agent = Agent(model=LlmClient(model=model), tools=mcp_tools,
              instructions=gaia_prompt, output_type=GaiaOutput, max_steps=15)
```

```typescript
const agent = new Agent({
  model: new LlmClient({ model }),
  tools: mcpTools,
  instructions: gaiaPrompt,
  outputType: GaiaOutput,
  maxSteps: 15,
});
```

裸 LLM 答不出的"需要当前信息"类问题，Agent 靠搜索工具解决了 —— **同一个模型，装上循环和工具，能力边界直接改变。** 这是全书主张的第一次实证。

调试利器 `display_trace(result.context)`（`scratch_agents/utils.py`）：因为 events 记录了一切，可以逐事件回放 Agent 的每一步思考、每次调用、每个结果。**可观测性是架构分离白送的礼物。**

---

## 本章要带走的 5 句话

1. ReAct = 思考→行动→观察的循环；现代实现用原生 tool calling，不再解析文本。
2. 架构的灵魂是**存储与呈现分离**：`context.events` 是唯一事实来源，`LlmRequest` 是每次现做的视图 —— 第 6 章的记忆优化全部长在这条缝上。
3. `Message/ToolCall/ToolResult` 是厂商中立的内部语言，OpenAI 格式被压缩在 `LlmClient` 的翻译边界上。
4. 错误处理哲学：工具异常、未知工具名、LLM 报错，全部变成历史里的 error 记录 —— **让模型看见错误并自我修正，而不是让程序崩溃**。
5. 结构化输出 = "final_answer 也是工具" + `tool_choice="required"` —— 复用工具机制做输出约束，这是框架设计里"新需求不加新机制"的典范。
