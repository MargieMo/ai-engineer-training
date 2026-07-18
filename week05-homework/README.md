# 第五周作业
## 基于MCP协议的多代理文章自动编写系统

### 项目概述
开发一个使用MCP (Model Context Protocol) 的多代理系统，能够协作完成文章写作任务。系统包含四个专业化代理，按顺序协作完成从研究到最终成稿的完整流程。

### 核心功能要求
- **输入**：用户问题（如"帮我写一篇关于AI Agent的文章"）
- **输出**：完成的文章文档和执行过程记录

### 系统架构
系统需要包含以下四个代理，按顺序协作：

1. **研究代理 (Research Agent)**
   - 使用搜索工具收集相关信息
   - 输出结构化的研究资料

2. **撰写代理 (Writing Agent)**
   - 基于研究结果生成文章初稿
   - 支持调整文章风格和长度

3. **审核代理 (Review Agent)**
   - 检查内容质量和逻辑一致性
   - 提供修改建议

4. **润色代理 (Polishing Agent)**
   - 优化语言表达和文章结构
   - 确保风格一致性

### 技术实现
- 可以使用现有的MCP库
- 代理间通过结构化消息进行通信
- 终端实时展示协作过程
- 生成: 示例输出文档（展示完整代理协作过程与最终成果）

### 扩展项（选做）
 - 实现基于MCP上下文的自动重试：当代理执行失败时，系统保留完整上下文并尝试替代方案 
   - 设置三级重试策略：
     * 一级：相同代理重新执行（最多2次）
     * 二级：切换至备用代理执行（如审核失败转由高级审核代理处理）
     * 三级：向用户请求补充信息
   - 所有重试过程需记录在最终文档的"异常处理日志"部分 

### 本项目实现

本目录已经完成核心要求和扩展项：

- 使用 LangGraph 编排 `研究 → 撰写 → 审核 → 润色` 四个代理
- 四个代理及备用代理均使用 OpenAI `gpt-5`
- 使用 FastMCP 提供 DuckDuckGo 搜索和集中式提示词服务
- 通过 `AgentState` 在代理间传递结构化上下文
- 实现自定义主题、文章风格和目标字数
- 撰写与润色 prompt 中使用目标字数作为软约束
- 润色阶段对终稿做硬性字数校验（目标字数的 85%–115%），超出则触发重试/备用代理
- 实现三级恢复：同代理重试两次、备用代理、请求用户补充信息
- 输出拆分为 `final_article_*.md` 与 `process_log_*.md`，过程日志不再重复粘贴终稿

架构如下：

```text
CLI / LangGraph client
  └─ researcher → writer → reviewer → polisher
          │           │          │          │
          └────────── MCP tools server ─────┘
                       ├─ search
                       └─ get_prompt
```

### 安装与运行

要求 Python 3.11+ 和 [uv](https://docs.astral.sh/uv/)。

```bash
cd week05-homework
uv sync
cp .env.example .env
```

在 `.env` 中填写 `OPENAI_API_KEY`。默认模型为 `gpt-5`，可以通过
`OPENAI_MODEL` 修改。

终端 1 启动 MCP 服务：

```bash
uv run python -m multi-agent.mcp_server.main
```

终端 2 启动写作客户端：

```bash
uv run python -m multi-agent.main
```

输出保存在本目录的两个文件中：

- `final_article_YYYYMMDD_HHMMSS.md`：仅包含最终交付文章
- `process_log_YYYYMMDD_HHMMSS.md`：包含研究、初稿、审核等过程记录和异常处理日志

如果三级恢复后仍然失败，只会生成 `process_log_YYYYMMDD_HHMMSS.md`。

运行自动重试测试：

```bash
uv run pytest
```


### 提交要求
在以下目录完成编码作业：
- [week05-homework/multi-agent](./multi-agent)

其中:
- `main.py` 是作业入口文件
- `report.md` 是示例输出文档（展示完整代理协作过程与最终成果）

### 提交方式
1. Fork本仓库
2. 在你的仓库中完成代码
3. 在【极客时间】提交fork仓库链接，格式为：
```
https://github.com/your-username/ai-engineer-training/tree/main/week05-homework
```