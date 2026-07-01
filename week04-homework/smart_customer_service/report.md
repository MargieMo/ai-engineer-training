# 智能客服系统实验结果分析报告（中文版）

> 英文版见 [`report_en.md`](./report_en.md)。

## 1. 项目概述

本项目实现了一个基于 LangChain 与 LangGraph 的智能客服系统，底层大模型使用 **OpenAI**（默认 `gpt-5`，通过 `langchain-openai` 的 `ChatOpenAI` 接入）。系统具备多轮对话管理、工具自动调用、模型与插件热更新等能力，能够处理查询订单、申请退款、开具发票以及相对时间解析等操作，并支持在不重启服务的情况下更新模型和工具。

## 2. 系统架构

系统按职责分为五层：

- **阶段一链 `base_chain.py`（Prompt → LLM → OutputParser）**：独立的 LCEL 链，结合锚定日期将「我昨天下的单」等相对时间推断为具体日期；可通过 `python -m smart_customer_service.main phase1` 单独运行。
- **服务层 `services.py`（`ServiceManager` 单例）**：统一持有 LLM 与工具集，是热更新的唯一数据源，提供 `update_llm()` 与 `update_tools()`。
- **编排层 `graph.py`（`GraphManager` + LangGraph 状态图）**：编排对话流程，使用 `MemorySaver` 检查点保存按会话（thread）隔离的对话记忆。
- **接口层 `api.py`（FastAPI）**：对外暴露 `/health`、`/chat`、`/hot-update` 三个端点。
- **入口 `main.py`**：`server` 启动 API，`phase1` 运行阶段一链 demo，`cli` 提供 LangGraph 终端多轮对话。

## 3. 核心功能实现与测试结果

### 3.1 阶段一：基础 Chain（相对时间推断）

`base_chain.py` 使用 LangChain LCEL 显式实现 **Prompt → LLM → OutputParser**：

- **Prompt**：将锚定日期（2025-09-26）与用户输入注入系统提示。
- **LLM**：OpenAI 模型解析「昨天」「上周三」等表达。
- **OutputParser**：`StrOutputParser` 输出自然语言形式的解析结果。

运行 `python -m smart_customer_service.main phase1` 可查看内置样例并交互测试。阶段二及以后的多轮对话仍通过 LangGraph + `get_date_for_relative_time` 工具处理相对时间。

### 3.2 对话系统功能

系统基于 LangGraph 构建了状态驱动的对话流程，包含以下节点：

- **agent 节点**：调用 OpenAI 模型（绑定工具），决定是否发起工具调用。
- **tools 节点**：执行具体工具，如查询订单、申请退款、开具发票、解析时间。
- **ask_for_order_id 节点**：当用户说"查订单"但未提供订单号时，主动追问订单号。

测试表明系统能正确识别意图：当用户查询订单但缺少订单号时，会主动追问；提到"昨天""上周三"等相对时间时会交由 agent 调用时间工具解析后再处理，体现了良好的多轮对话管理能力。

### 3.3 工具调用功能

系统实现了四个工具：

1. **query_order**：根据订单号查询订单状态与物流信息（内置 mock 数据）。
2. **apply_refund**：为指定订单申请退款，返回退款单号。
3. **generate_invoice**：为指定订单生成发票下载链接。
4. **get_date_for_relative_time**：将"昨天""前天""上周三"（及英文 yesterday/last Wednesday 等）转换为 `YYYY-MM-DD` 具体日期。

自动化测试验证了发票工具能生成包含订单号的链接、非法订单号能优雅失败，以及相对时间解析结果正确（相对于 2025-09-26，"昨天"= 2025-09-25）。

### 3.4 热更新功能

系统支持两类热更新：

1. **模型热更新**：运行时切换 OpenAI 模型（如 `gpt-5` → `gpt-4o`）。
2. **工具热更新**：运行时替换可用工具集（如仅保留查询工具）。

热更新后调用 `reload_graph()` 重建图实例，使新会话使用新配置，而正在进行的旧会话可完成其生命周期。结合 LangGraph 的 `thread_id` 与 `MemorySaver`，实现了会话状态隔离，保证服务连续性。

## 4. 快速开始

```bash
cd week04-homework
uv sync                       # 安装依赖
cp .env.example .env          # 填入 OPENAI_API_KEY

# 方式一：阶段一 Chain demo
python -m smart_customer_service.main phase1

# 方式二：启动 API 服务
python -m smart_customer_service.main server
# 健康检查
curl http://localhost:8000/health
# 对话
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"user_id":"u1","query":"我想查订单"}'
# 热更新模型
curl -X POST http://localhost:8000/hot-update -H "Content-Type: application/json" \
  -d '{"type":"model","name":"gpt-5"}'

# 方式三：终端交互模式
python -m smart_customer_service.main cli

# 运行测试
python -m unittest tests.test_features -v
```

## 5. 性能与稳定性分析

- **响应时间**：主要消耗在 LLM 调用（约 1–3 秒）与工具模拟延迟（约 1–2 秒），系统自身处理开销可忽略。
- **稳定性保障**：模型调用处加了异常捕获避免崩溃；`/health` 端点用于监控；通过 `thread_id` + 检查点实现会话隔离。

## 6. 存在的问题与改进方向

- **意图识别简单**：目前入口路由基于关键词匹配，可替换为专门的意图识别模型。
- **工具为 mock 实现**：订单/退款/发票均为模拟数据，实际需接入真实后端与数据库。
- **记忆为内存实现**：`MemorySaver` 重启即丢失，生产环境建议使用持久化检查点（如数据库）。
- **错误处理可完善**：可增加更多异常场景的重试与降级策略。

## 7. 总结

本项目成功实现了一个功能完整、以 OpenAI 为底座的智能客服系统，覆盖作业要求的三个阶段：基础对话与相对时间推断、LangGraph 多轮对话与工具调用、以及热更新 / 健康检查 / 自动化测试。系统结构清晰、易于扩展，热更新机制使其具备生产环境所需的可维护性。
