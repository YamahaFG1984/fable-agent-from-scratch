# 第 2 章：LLM API 基础（教学笔记）

> 对应 notebook：`notebooks/ch02/ch02_llm_api_basics.ipynb`
> 核心代码：`scratch_agents/eval/gaia.py`

这一章回答一个问题：**在写任何 "Agent" 代码之前，你必须先掌握怎么和 LLM 对话。** 它由 9 个递进的知识点组成，最后用一个真实基准测试收尾。

## 1. 最原始的调用：OpenAI 与 Anthropic API（Listing 2.2–2.3）

两家 API 的共同骨架是：**一个模型名 + 一个 messages 列表 → 一个回复**。

```python
# OpenAI
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-5.4-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
)
print(response.choices[0].message.content)
```

```python
# Anthropic
from anthropic import Anthropic
client = Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "What is the capital of France?"}]
)
print(response.content[0].text)
```

**要理解的核心是 `role` 的三种身份**：

- `system` —— 开发者给模型的行为设定（Anthropic 是单独的 `system` 参数，不放 messages 里）
- `user` —— 用户输入
- `assistant` —— 模型的历史回复

注意两家的差异：字段名不同（`choices[0].message.content` vs `content[0].text`）、Anthropic 强制要求 `max_tokens`。这个"差异之痛"直接引出下一节。

## 2. 统一接口：LiteLLM（Listing 2.4）

```python
from litellm import completion

response = completion(model="gpt-5.4-mini", messages=[...])       # OpenAI
response = completion(model="claude-sonnet-4-6", messages=[...])  # Anthropic
```

LiteLLM 把所有厂商适配成 **OpenAI 格式**，换模型只需换字符串。这是一个重要的工程决策：本书整个框架的 LLM 层（`scratch_agents/llm.py` 里的 `LlmClient`）就构建在 `litellm.acompletion` 之上，所以第 4 章写好的 Agent 天然支持任何模型。

## 3. 全章最重要的概念：API 是无状态的（Listing 2.5）

```python
# 第一次调用
completion(model=..., messages=[{"role": "user", "content": "My name is Jungjun."}])

# 第二次调用
completion(model=..., messages=[{"role": "user", "content": "What is my name?"}])
# → 模型答不出来！
```

**LLM API 没有记忆。** 每次调用都是全新的，服务器不保存你上一轮说了什么。第二次调用时模型完全不知道你叫 Jungjun。

这不是缺陷，而是设计 —— 它把"记忆"的责任完全交给了**你（客户端）**。

## 4. 解法：自己维护对话历史（Listing 2.6）

```python
messages = []

messages.append({"role": "user", "content": "My name is Jungjun."})
response1 = completion(model="gpt-5.4-mini", messages=messages)
messages.append({"role": "assistant", "content": response1.choices[0].message.content})

messages.append({"role": "user", "content": "What is my name?"})
response2 = completion(model="gpt-5.4-mini", messages=messages)
# → "Your name is Jungjun." ✓
```

模式就是：**每轮把用户输入和模型回复都 append 进列表，下次把整个列表重新发过去。** 所谓"模型记住了你"，本质是你每次都把全部历史重新喂给它。

**这一个模式是全书的种子**，后面几乎每一章都是它的延伸：

- 第 4 章的 ReAct 循环 = 把工具调用/结果也 append 进历史再重发（`ExecutionContext.events` 就是这个 `messages` 列表的结构化升级版）
- 第 6 章的记忆系统 = 历史太长塞不下时怎么裁剪、压缩、总结

## 5. 结构化输出：让 LLM 返回数据而不是散文（Listing 2.7）

```python
from pydantic import BaseModel

class ExtractedInfo(BaseModel):
    name: str
    email: str
    phone: str | None = None

response = completion(
    model="gpt-5.4-mini",
    messages=[{"role": "user", "content": "My name is John Smith, my email is john@example.com..."}],
    response_format=ExtractedInfo,   # ← 关键
)
```

传入一个 Pydantic 模型作为 `response_format`，模型就被约束输出符合该 schema 的 JSON。**这是把 LLM 从"聊天机器人"变成"软件组件"的关键**——程序无法可靠地解析自由文本，但能可靠地解析 `ExtractedInfo`。第 4 章 Agent 的 `output_type` 参数、第 6 章的记忆抽取（`TaskMemory`），都靠这一招。

## 6. 异步并发调用（Listing 2.8）

```python
import asyncio
from litellm import acompletion  # 注意 a 前缀 = async 版本

async def get_response(prompt: str) -> str:
    response = await acompletion(model="gpt-5.4-mini", messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content

tasks = [get_response(p) for p in prompts]
results = await asyncio.gather(*tasks)   # 三个请求同时飞出去
```

LLM 调用的瓶颈是**网络等待**而不是本地计算，所以适合异步：三个请求并发，总耗时约等于最慢的那一个，而不是三者之和。这也是为什么整个框架的 `Agent.run()`、工具执行全部是 `async` 的。

## 7. 并发限流 + 自动重试（Listing 2.9）

放开并发马上会撞上现实：**API 有速率限制（rate limit）**。解法是信号量：

```python
semaphore = asyncio.Semaphore(10)   # 最多 10 个请求同时在飞

async def call_llm(prompt: str) -> str:
    async with semaphore:           # 第 11 个请求会排队等
        response = await acompletion(
            model=...,
            messages=[...],
            num_retries=3,          # 失败时指数退避自动重试
        )
        return ...

results = await asyncio.gather(*tasks, return_exceptions=True)  # 单个失败不炸全局
```

三个生产级细节：`Semaphore` 控并发、`num_retries` 抗瞬时故障、`return_exceptions=True` 让一个失败不中断其余 99 个。

## 8. 实战收尾：GAIA 基准测试（Listing 2.10–2.17）

学了这些之后，本章用一个真实任务把它们全部串起来：**用 GAIA 基准测量"裸 LLM"到底有多强**。GAIA 是专门为评估 AI 助手设计的题库，题目需要推理、查资料、处理文件。

```python
from datasets import load_dataset
level1_problems = load_dataset("gaia-benchmark/GAIA", "2023_level1", split="validation")

from scratch_agents.eval.gaia import run_experiment
MODELS = ["gpt-5.5", "gpt-5.4-mini", "anthropic/claude-sonnet-4-6", "anthropic/claude-haiku-4-5"]
results = await run_experiment(level1_problems.select(range(20)), MODELS)
```

实现在 `scratch_agents/eval/gaia.py`，里面能看到本章每个知识点的落地：

| 本章知识点 | 在 gaia.py 中的体现 |
|-----------|-------------------|
| 统一 API | `litellm.acompletion` 跑通 4 个不同厂商模型 |
| 结构化输出 | `GaiaOutput(is_solvable, unsolvable_reason, final_answer)` |
| 异步并发 | `tqdm_asyncio.gather(*tasks)` 所有题×所有模型一起跑 |
| 限流 | 按厂商分别限流：`{"openai": Semaphore(30), "anthropic": Semaphore(10)}` |
| 重试 | `num_retries=2` |

有个值得注意的提示词设计：让模型自己申报 `is_solvable`（能不能解）。裸 LLM 面对"打开这个 Excel 文件统计销量"这类题只能说"我做不到"—— **这个失败就是全书的引子**：模型缺的不是智力，是"手"。第 3 章给它工具，第 4 章教它自主循环使用工具，成为真正的 Agent。

---

## 本章要带走的 5 句话

1. LLM API = `messages` 进，一条回复出；`role` 区分系统/用户/助手。
2. **API 无状态** —— 记忆是假象，是客户端每次重发全部历史造出来的。这是理解 Agent 的第一性原理。
3. 用 LiteLLM 统一多厂商，用 Pydantic `response_format` 把 LLM 变成可靠的程序组件。
4. 生产级调用三件套：异步并发、信号量限流、自动重试。
5. 裸 LLM 在 GAIA 上的失败证明：**它需要工具和执行循环** —— 这正是后续章节要构建的东西。
