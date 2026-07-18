"""Entry point: run one full werewolf game and write a replay log."""

from __future__ import annotations

import sys

from . import config
from .graph import build_graph, create_initial_state
from .logger import write_replay


def main() -> None:
    if not config.OPENAI_API_KEY:
        print("错误：未配置 OPENAI_API_KEY，请复制 .env.example 为 .env 并填写。")
        sys.exit(1)

    print("=" * 60)
    print("狼人杀多智能体对局启动")
    print(f"模型: {config.OPENAI_MODEL}")
    print(f"最大轮次: {config.MAX_ROUNDS}")
    print("=" * 60)

    graph, _memory = build_graph()
    state = create_initial_state(max_rounds=config.MAX_ROUNDS)

    # Stream node updates so the console shows progress.
    final_state = state
    for update in graph.stream(state, stream_mode="updates"):
        for node_name, node_state in update.items():
            final_state = {**final_state, **node_state}
            phase = node_state.get("phase", node_name)
            round_no = node_state.get("round", "?")
            print(f"[R{round_no}] 完成节点: {node_name} (phase={phase})")
            if node_state.get("public_log"):
                # Print only newly appended public lines when possible.
                pass
            if node_name == "dawn_announce":
                for line in node_state.get("public_log", [])[-2:]:
                    print(f"  {line}")
            if node_name == "day_vote":
                for line in node_state.get("public_log", [])[-1:]:
                    print(f"  {line}")
            if node_state.get("winner"):
                print(f"  >>> 胜者: {node_state['winner']}")

    path = write_replay(final_state)
    print("=" * 60)
    print(f"对局结束。完整回放已写入: {path}")
    print(f"Total tokens: {final_state['cost']['total_tokens']}")
    print(f"LLM calls: {final_state['cost']['calls']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
