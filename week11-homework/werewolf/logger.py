"""Write full game replay markdown logs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from . import config
from .models import GameState
from .tracing import summarize_cost


def write_replay(state: GameState) -> Path:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = config.LOG_DIR / f"game_replay_{stamp}.md"

    role_lines = []
    for name, meta in state["players"].items():
        status = "存活" if meta["alive"] else "出局"
        role_cn = "狼人" if meta["role"] == "werewolf" else "村民"
        role_lines.append(
            f"- {name}: {role_cn} / {meta['personality']} / {status}"
        )

    winner = state.get("winner") or "未分胜负"
    winner_cn = {
        "villagers": "村民阵营",
        "werewolves": "狼人阵营",
    }.get(winner, winner)

    trace_blocks = []
    for t in state["traces"]:
        trace_blocks.append(
            "\n".join(
                [
                    f"#### R{t['round']} · {t['phase']} · {t['agent']}",
                    f"- **Thought**: {t['thought']}",
                    f"- **Action**: {t['action']}",
                    f"- **Observation**: {t['observation']}",
                ]
            )
        )

    content = "\n".join(
        [
            "# 狼人杀对局回放",
            "",
            f"- 时间: {stamp}",
            f"- 模型: {config.OPENAI_MODEL}",
            f"- 结束轮次: 第 {state['round']} 轮",
            f"- 胜者: {winner_cn}",
            "",
            "## 角色复盘（开局身份）",
            "",
            *role_lines,
            "",
            "## 完整对局日志",
            "",
            *state["public_log"],
            "",
            "## 执行追踪（Thought / Action / Observation）",
            "",
            *trace_blocks,
            "",
            "## 成本与复杂度分析",
            "",
            summarize_cost(state["cost"]),
            "",
            "### 复杂度说明",
            "",
            "- 每轮固定阶段：夜晚行动 → 天亮公布 → 发言 → 投票 → 胜负判定",
            "- LLM 调用规模约 O(轮次 × 存活玩家)，另加狼人夜间决策",
            "- RAG：每条发言写入 FAISS；发言/投票前 Top-K 语义召回",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")
    return path
