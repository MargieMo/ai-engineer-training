"""Streamlit UI: live werewolf game flow + Thought/Action/Observation."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from . import config
from .graph import PLAYER_SETUP, build_graph, create_initial_state
from .logger import write_replay
from .tracing import summarize_cost

PHASE_LABELS = {
    "night_action": "夜晚行动",
    "dawn_announce": "天亮公布",
    "day_speak": "发言环节",
    "day_vote": "投票处决",
    "check_win": "胜负判定",
}

ROLE_CN = {"werewolf": "狼人", "villager": "村民"}
PERSONALITY_CN = {
    "aggressive": "激进",
    "cautious": "谨慎",
    "analytical": "分析",
}


def _init_session() -> None:
    defaults = {
        "running": False,
        "final_state": None,
        "node_history": [],
        "replay_path": None,
        "error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_players(state: dict, *, god_view: bool) -> None:
    cols = st.columns(len(PLAYER_SETUP))
    for col, (name, role, personality) in zip(cols, PLAYER_SETUP):
        meta = state.get("players", {}).get(name, {})
        alive = meta.get("alive", True)
        status = "存活" if alive else "出局"
        with col:
            st.markdown(f"**{name}**")
            st.caption(status)
            if god_view:
                st.write(
                    f"{ROLE_CN[role.value]} · {PERSONALITY_CN[personality.value]}"
                )
            else:
                st.write(f"性格倾向：{PERSONALITY_CN[personality.value]}")
            if not alive:
                st.markdown(":red[已出局]")


def _render_timeline(node_history: list[dict]) -> None:
    if not node_history:
        st.info("尚未开始。点击侧边栏「开始对局」。")
        return
    for item in node_history:
        label = PHASE_LABELS.get(item["node"], item["node"])
        with st.expander(
            f"R{item['round']} · {label} · phase={item['phase']}",
            expanded=(item is node_history[-1]),
        ):
            if item.get("snippet"):
                st.markdown(item["snippet"])


def _render_traces(traces: list[dict], limit: int = 40) -> None:
    if not traces:
        st.caption("暂无追踪记录")
        return
    for t in reversed(traces[-limit:]):
        with st.container(border=True):
            st.markdown(
                f"**R{t['round']} · {t['phase']} · {t['agent']}**"
            )
            st.markdown(f"- Thought：{t['thought']}")
            st.markdown(f"- Action：{t['action']}")
            st.markdown(f"- Observation：{t['observation']}")


def _snippet_for_node(node_name: str, state: dict) -> str:
    log = state.get("public_log") or []
    if node_name == "dawn_announce":
        return "\n\n".join(log[-2:])
    if node_name == "day_speak":
        speeches = [
            line for line in log if line.startswith("**") and "投票" not in line
        ]
        return "\n\n".join(speeches[-5:]) if speeches else "（本轮发言已完成）"
    if node_name == "day_vote":
        return "\n\n".join(log[-6:])
    if node_name == "check_win":
        return log[-1] if log else ""
    if node_name == "night_action":
        return "狼人正在选择击杀目标…"
    return ""


def run_game(max_rounds: int, god_view: bool) -> None:
    st.session_state.running = True
    st.session_state.error = None
    st.session_state.node_history = []
    st.session_state.final_state = None
    st.session_state.replay_path = None

    status = st.status("对局进行中…", expanded=True)
    players_box = st.empty()
    timeline_box = st.empty()
    traces_box = st.empty()
    log_box = st.empty()

    try:
        graph, _memory = build_graph()
        state = create_initial_state(max_rounds=max_rounds)
        final_state = state

        with status:
            st.write("图已编译，开始夜晚行动。")

        for update in graph.stream(state, stream_mode="updates"):
            for node_name, node_state in update.items():
                final_state = node_state
                history_item = {
                    "node": node_name,
                    "round": node_state.get("round", 0),
                    "phase": node_state.get("phase", node_name),
                    "snippet": _snippet_for_node(node_name, node_state),
                }
                st.session_state.node_history.append(history_item)

                label = PHASE_LABELS.get(node_name, node_name)
                with status:
                    st.write(
                        f"完成：R{history_item['round']} · {label}"
                    )

                with players_box.container():
                    st.subheader("玩家状态")
                    _render_players(final_state, god_view=god_view)

                with timeline_box.container():
                    st.subheader("阶段时间线")
                    _render_timeline(st.session_state.node_history)

                with traces_box.container():
                    st.subheader("执行追踪（最新在上）")
                    _render_traces(final_state.get("traces") or [])

                with log_box.container():
                    st.subheader("公开对局日志")
                    st.markdown("\n\n".join(final_state.get("public_log") or []))

        path = write_replay(final_state)
        st.session_state.final_state = final_state
        st.session_state.replay_path = str(path)
        winner = final_state.get("winner") or "unknown"
        status.update(
            label=f"对局结束 · 胜者：{winner}",
            state="complete",
        )
    except Exception as exc:  # noqa: BLE001 - surface to UI
        st.session_state.error = str(exc)
        status.update(label="对局失败", state="error")
        st.exception(exc)
    finally:
        st.session_state.running = False


def _load_replay_file(path: Path) -> None:
    st.markdown(path.read_text(encoding="utf-8"))


def main() -> None:
    st.set_page_config(
        page_title="狼人杀 Agent 可视化",
        page_icon="🐺",
        layout="wide",
    )
    _init_session()

    st.title("狼人杀多智能体 · 执行流可视化")
    st.caption(
        f"模型：{config.OPENAI_MODEL or '未配置'} · "
        "展示阶段流转与 Thought / Action / Observation"
    )

    with st.sidebar:
        st.header("控制台")
        max_rounds = st.slider("最大轮次", 1, 8, config.MAX_ROUNDS)
        god_view = st.toggle("上帝视角（显示真实身份）", value=True)
        start = st.button(
            "开始对局",
            type="primary",
            disabled=st.session_state.running or not config.OPENAI_API_KEY,
            use_container_width=True,
        )
        if not config.OPENAI_API_KEY:
            st.error("未配置 OPENAI_API_KEY")

        st.divider()
        st.subheader("历史回放")
        log_dir = config.LOG_DIR
        replays = sorted(log_dir.glob("game_replay_*.md"), reverse=True) if log_dir.exists() else []
        replay_names = [p.name for p in replays]
        selected = st.selectbox("选择日志", options=["（不加载）", *replay_names])
        load_replay = st.button("加载选中回放", use_container_width=True)

    if start:
        run_game(max_rounds=max_rounds, god_view=god_view)

    if st.session_state.error:
        st.error(st.session_state.error)

    if st.session_state.final_state and not start:
        state = st.session_state.final_state
        left, right = st.columns([1.1, 1])
        with left:
            st.subheader("玩家状态")
            _render_players(state, god_view=god_view)
            st.subheader("阶段时间线")
            _render_timeline(st.session_state.node_history)
            st.subheader("成本统计")
            st.markdown(summarize_cost(state["cost"]))
            if st.session_state.replay_path:
                st.success(f"回放已保存：`{st.session_state.replay_path}`")
        with right:
            st.subheader("执行追踪")
            _render_traces(state.get("traces") or [])
            st.subheader("公开对局日志")
            st.markdown("\n\n".join(state.get("public_log") or []))

    if load_replay and selected != "（不加载）":
        path = config.LOG_DIR / selected
        st.subheader(f"回放：{selected}")
        _load_replay_file(path)

    if not start and not st.session_state.final_state and selected == "（不加载）":
        st.markdown(
            """
### 使用说明
1. 在侧边栏设置最大轮次，可选打开上帝视角  
2. 点击 **开始对局**，页面会实时刷新阶段与追踪  
3. 结束后可在侧边栏加载 `logs/game_replay_*.md` 回看  

也可用命令行跑无界面版本：
```bash
uv run python -m werewolf.main
```
"""
        )


if __name__ == "__main__":
    main()
