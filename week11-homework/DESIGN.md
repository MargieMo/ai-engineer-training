# 狼人杀智能体系统 — 设计文档

## 1. 架构选择

选用 **LangChain + LangGraph**：

| 需求 | 对应能力 |
|------|----------|
| 夜晚→天亮→发言→投票循环 | LangGraph 状态机精确控流 |
| 记忆 + RAG | LangChain Embeddings + FAISS 易集成 |
| 可观测 | 节点内统一记录 Thought / Action / Observation |

整体结构：

```
Moderator (LangGraph) ──调度──► Player Agents (LLM)
        │                              │
        ▼                              ▼
  GameState / 规则校验          Episodic + Semantic Memory
        │                              │
        └────────► Trace / Cost Logger ◄┘
```

## 2. 角色与 Prompt

- **2 狼人**：`aggressive`（激进带节奏）、`cautious`（谨慎伪装）
- **3 村民**：`analytical`（逻辑分析）、`aggressive`（激进指控）、`cautious`（谨慎观望）

每个 Agent 注入：公开身份（玩家名）、私有身份（狼/村）、性格、队友信息（仅狼人）、当前局势摘要、RAG 召回的历史发言。

## 3. 游戏流程（LangGraph）

```
START → night_action → dawn_announce → day_speak → day_vote → check_win
              ▲                                                    │
              └────────────── 未分胜负 ────────────────────────────┘
                                                                   │
                                                              已分胜负 → END
```

- **night_action**：存活狼人协商击杀目标（主持人校验）
- **dawn_announce**：公布昨夜死亡；无人死亡则平安夜
- **day_speak**：存活玩家依次发言（可检索历史）
- **day_vote**：存活玩家投票；平票由主持人随机处决一人
- **check_win**：狼人人数 > 村民 → 狼人胜；狼人全灭 → 村民胜；否则进入下一夜
  （5 人局用严格大于，避免首夜击杀后 2v2 立刻结束、跑不出白天发言）

最多进行 `MAX_ROUNDS` 轮，超时按存活人数判定。

## 4. 记忆与 RAG

### 情景记忆（Episodic）

结构化事件列表：死亡、发言、投票、夜晚行动。发言前拼进 Prompt 作为「局势摘要」。

### 语义记忆（Semantic / RAG）

- 将每条公开发言写入 **FAISS** 向量库（`OpenAIEmbeddings`）
- 发言 / 投票前，按当前局势 query 召回 Top-K 相似历史发言
- Agent 用召回内容做反驳或佐证，体现 RAG 在决策链中的作用

## 5. 调试与可观测性

每步记录：

- **Thought**：局势判断与策略意图
- **Action**：发言 / 投票 / 击杀目标
- **Observation**：主持人反馈（是否合法、公布结果）

日志写入 `logs/game_replay_*.md`，并附 token / 延迟统计。

## 6. 成本分析（附加）

运行结束统计：

- 总 LLM 调用次数、prompt/completion/total tokens
- 平均响应延迟
- 粗算 GPU 资源：按调用次数与模型体量给出本地部署量级估算

## 7. 可视化界面（Streamlit）

本地原型：`streamlit_app.py` / `werewolf/app.py`

- 侧边栏启动对局，实时刷新阶段时间线
- 展示玩家状态、公开日志、Thought / Action / Observation
- 支持上帝视角开关与历史 `logs/game_replay_*.md` 回放加载

```bash
cd week11-homework
uv sync
uv run streamlit run streamlit_app.py
```
