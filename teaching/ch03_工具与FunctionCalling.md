# 第 3 章：让 LLM 行动起来 —— 工具使用（Tool Use / Function Calling）（教学笔记）

> 对应 notebook：`notebooks/ch03/ch03_tools_and_function_calling.ipynb`（Listing 3.1–3.26）
> 核心代码：`scratch_agents/tools/helpers.py`、`tools/calculator.py`、`tools/search.py`、`tools/mcp.py`

上一章的结论是：裸 LLM 只有"脑"没有"手"。本章就是给它装手 —— 但第一个要建立的观念恰恰是：**手不是模型的，是你的。**

## 1. 工具调用的真相：LLM 从不执行任何东西（Listing 3.1–3.4）

初学者最大的误解是"模型会调用工具"。真实机制是一场**结构化的问答游戏**：

```
你 ──(问题 + 工具清单)──→ LLM
LLM ──("请帮我执行 calculator(multiply, 1234, 5678)")──→ 你
你自己执行函数 → 把结果发回去
LLM ──("答案是 7,006,652")──→ 你
```

模型只做一件事：**输出一段 JSON，声明它想调用什么、参数是什么**。执行、安全、错误处理，全在你的代码里。

### 工具的两半：定义 + 实现

**给模型看的一半 —— 工具定义（JSON Schema）：**

```python
calculator_tool_definition = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Perform basic arithmetic operations.",  # 模型靠这句话决定何时用它
        "parameters": {
            "type": "object",
            "properties": {
                "operator": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                "first_number": {"type": "number"},
                "second_number": {"type": "number"},
            },
            "required": ["operator", "first_number", "second_number"],
        },
    },
}
```

**给 Python 跑的一半 —— 普通函数：** `def calculator(operator, first_number, second_number): ...`

两者靠 `name` 关联。模型看到的只有 schema，永远碰不到代码。

### 模型自己决定用不用工具（Listing 3.3 对照实验）

- 问 *"韩国的首都是哪？"* → `tool_calls` 为 `None`，直接回答
- 问 *"1234 × 5678？"* → `content` 为 `None`，`tool_calls` 里出现 calculator 请求

这就是"自主性"的最小形态：**你只提供能力，用不用、怎么用由模型判断。**

### 解析并执行

```python
for tool_call in ai_message.tool_calls:
    function_args = json.loads(tool_call.function.arguments)  # arguments 是 JSON 字符串！
    result = calculator(**function_args)
```

`arguments` 是模型生成的**文本**，可能畸形，生产代码要处理解析失败。

## 2. 完整闭环：把结果喂回去（Listing 3.5）

```python
messages = [{"role": "user", "content": "What is 1234 x 5678?"}]

# A. 追加 assistant 的工具调用消息（原样保留 tool_calls）
messages.append({"role": "assistant", "content": ai_message.content,
                 "tool_calls": ai_message.tool_calls})
# B. 执行工具
result = calculator(**function_args)
# C. 用 role="tool" 追加结果，tool_call_id 负责配对
messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(result)})

final_response = completion(model="gpt-5.4-mini", messages=messages)
```

两条规则：

1. **第四种角色 `tool`**（前三种是 system/user/assistant），专门装工具结果。
2. **`tool_call_id` 是配对凭证**；`assistant(tool_calls)` 后必须紧跟对应 `tool` 消息，顺序错了 API 报错。

工具结果能被模型"知道"，还是因为你把它 append 进历史重发了 —— 第 2 章无状态原理的新用法。

## 3. 工具设计准则：描述就是提示词工程（Listing 3.6–3.11）

三组反模式：

**① 模糊 vs 明确**：`book / "Book something"` 模型没法用；`reserve_table` + ISO-8601 格式说明 + `minimum/maximum` 约束才可靠。

**② 互相冲突的参数**：两个布尔 `on`/`off` 可同时为 true；改成一个 `enum: ["on", "off"]`，非法状态在 schema 层面就不存在。

**③ 让模型提供标识符（最危险）**：要求 `order_id` → 模型会**编造**一个像样的 ID，直接退错单。正确做法：ID 由系统上下文注入，模型只填决策类参数（`reason`）。

> 总原则：**模型只填它能从对话中真实获得的信息；系统性、身份性数据由代码注入。** 第 4 章框架中落地为 `context` 参数自动注入、不暴露给 LLM。

## 4. 构建真实工具：web 搜索（Listing 3.12–3.15）

三个版本的迭代，每版一条经验：

```python
# v3 最终版（= 框架 tools/search.py）
def search_web(query: str, max_results: int = 5,
               topic: str = "general", time_range: str | None = None) -> list | str:
    """Search the web for the given query."""
    try:
        response = tavily_client.search(query, max_results=max_results,
                                        topic=topic, time_range=time_range)
        return response.get("results")
    except Exception as e:
        return f"Error: Search failed - {e}"   # ← 关键！
```

- v1→v2：加可选参数，让模型能表达"搜最近一周的新闻"。
- v2→v3：**错误不抛异常而是返回字符串** —— 异常炸循环，错误文本进历史，模型看到后能自我调整重试。失败信息也是喂给模型的信息。

## 5. 自动生成工具定义：用内省消灭重复劳动（Listing 3.16–3.18）

```python
def function_to_input_schema(func) -> dict:
    type_map = {str: "string", int: "integer", float: "number", bool: "boolean", ...}
    signature = inspect.signature(func)
    parameters = {p.name: {"type": type_map.get(p.annotation, "string")}
                  for p in signature.parameters.values()}
    required = [p.name for p in signature.parameters.values()
                if p.default == inspect._empty]      # 没有默认值 = 必填
    return {"type": "object", "properties": parameters, "required": required}

def function_to_tool_definition(func) -> dict:
    return format_tool_definition(func.__name__, func.__doc__ or "",
                                  function_to_input_schema(func))
```

映射关系：**函数名→name，docstring→description，类型注解→参数类型，默认值有无→required**。"把注解和 docstring 写好"从代码规范变成了直接影响模型行为的提示词。这段代码原封不动进了 `scratch_agents/tools/helpers.py`，第 4 章 `@tool` 装饰器是它的封装。

## 6. 本书的第一个 Agent 雏形（Listing 3.19–3.21）⭐

```python
def simple_agent_loop(system_prompt, question):
    tools = [search_web]
    tool_box = {t.__name__: t for t in tools}
    tool_definitions = [function_to_tool_definition(t) for t in tools]
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": question}]

    while True:
        response = completion(model="gpt-5.4-mini", messages=messages,
                              tools=tool_definitions)
        assistant_message = response.choices[0].message
        if assistant_message.tool_calls:
            messages.append(assistant_message)
            for tool_call in assistant_message.tool_calls:
                tool_result = tool_execution(tool_box, tool_call)
                messages.append({"role": "tool", "content": str(tool_result),
                                 "tool_call_id": tool_call.id})
            # 不 return —— 回到循环顶部，让模型看到结果继续决策
        else:
            return assistant_message.content      # 没有工具调用 = 最终答案
```

与第 4 章 Agent 类的对应：

| simple_agent_loop | 第 4 章 Agent 类 |
|---|---|
| `while True` | `run()` 步数循环（多了 `max_steps` 保险） |
| `completion(...)` | `think()` |
| 执行工具 + append | `act()` |
| `messages` 列表 | `ExecutionContext.events` |
| "没有 tool_calls = 结束" | `_is_final_response()` 同一判据 |

缺陷（无步数上限、无错误恢复、状态与逻辑耦合）正是第 4 章的需求清单。

## 7. MCP：工具的标准化协议（Listing 3.22–3.26）

**MCP（Model Context Protocol）** 把工具做成独立"服务器"进程，任何 Agent 即插即用 —— 工具界的 USB 接口。

**客户端三步协议：initialize（握手）→ list_tools（发现）→ call_tool（调用）**，通信走子进程 stdio：

```python
server_params = StdioServerParameters(command="npx", args=["-y", "tavily-mcp@latest"],
                                      env={"TAVILY_API_KEY": ...})
async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        result = await session.call_tool("tavily-search", arguments={"query": "..."})
```

MCP 工具定义与 OpenAI 格式几乎同构（`inputSchema` → `parameters`），一个映射函数就能接进现有循环（框架 `tools/mcp.py`）。

**用 FastMCP 写服务器只要一个装饰器：**

```python
mcp = FastMCP("custom-tavily-search")

@mcp.tool()                      # 签名 + docstring 自动生成定义，与第 5 节同原理
def search_web(query: str, max_results: int = 5) -> str:
    """Search the web using Tavily API. ..."""
    ...

mcp.run(transport="stdio")
```

FastMCP 做的事和你手写的 `function_to_tool_definition` 一模一样 —— **标准协议只是把你已理解的机制固定下来。**

---

## 本章要带走的 5 句话

1. **LLM 永远不执行工具** —— 它只输出"调用意图"（JSON），执行权和责任在你的代码里。
2. 闭环四步：发工具清单 → 收 `tool_calls` → 执行并以 `role="tool"` + `tool_call_id` 回填 → 模型给最终答案。
3. 工具定义是提示词工程：描述要具体、用 enum 消灭非法状态、**永远别让模型编造 ID**；工具出错要返回错误文本而不是抛异常，让模型自我修正。
4. `inspect` 内省（函数名/docstring/类型注解 → 工具定义）消灭了手写 schema，这就是 `@tool` 装饰器和 FastMCP 的底层原理。
5. `simple_agent_loop` 的 20 行 `while True` 就是 Agent 的本质：**循环直到模型不再要求工具**。第 4 章只是给它加上结构、安全和扩展性。
