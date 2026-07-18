"""LangGraph moderator workflow for werewolf game phases."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from .agents import PlayerAgent
from .memory import GameMemory
from .models import GameState, Personality, Role
from .tracing import add_trace, make_llm


PLAYER_SETUP: list[tuple[str, Role, Personality]] = [
    ("Alice", Role.WEREWOLF, Personality.AGGRESSIVE),
    ("Bob", Role.WEREWOLF, Personality.CAUTIOUS),
    ("Carol", Role.VILLAGER, Personality.ANALYTICAL),
    ("Dave", Role.VILLAGER, Personality.AGGRESSIVE),
    ("Eve", Role.VILLAGER, Personality.CAUTIOUS),
]


def create_initial_state(max_rounds: int) -> GameState:
    players: dict[str, dict[str, Any]] = {}
    werewolves: list[str] = []
    villagers: list[str] = []
    for name, role, personality in PLAYER_SETUP:
        players[name] = {
            "name": name,
            "role": role.value,
            "personality": personality.value,
            "alive": True,
        }
        if role == Role.WEREWOLF:
            werewolves.append(name)
        else:
            villagers.append(name)

    from .tracing import empty_cost

    return {
        "round": 1,
        "phase": "night",
        "players": players,
        "alive": [p[0] for p in PLAYER_SETUP],
        "werewolves": werewolves,
        "villagers": villagers,
        "night_victim": "",
        "speeches": [],
        "votes": [],
        "eliminated": [],
        "winner": "",
        "traces": [],
        "public_log": [
            "【开局】5 名玩家入座：Alice, Bob, Carol, Dave, Eve",
            "【身份分配完成】（身份对玩家保密，仅日志末尾复盘公开）",
        ],
        "cost": empty_cost(),
        "max_rounds": max_rounds,
    }


class ModeratorNodes:
    def __init__(self, memory: GameMemory) -> None:
        self.memory = memory
        self.agent = PlayerAgent(make_llm(), memory)

    def _refresh_alive(self, state: GameState) -> None:
        state["alive"] = [
            name
            for name, meta in state["players"].items()
            if meta["alive"]
        ]

    def _eliminate(self, state: GameState, name: str, reason: str) -> None:
        if not name or name not in state["players"]:
            return
        if not state["players"][name]["alive"]:
            return
        state["players"][name]["alive"] = False
        state["eliminated"].append(name)
        self._refresh_alive(state)
        msg = f"{reason}：{name} 出局"
        state["public_log"].append(msg)
        self.memory.add_event(msg)

    def night_action(self, state: GameState) -> GameState:
        state["phase"] = "night"
        state["public_log"].append(f"\n## 第 {state['round']} 夜")
        state["public_log"].append("天黑请闭眼。狼人请睁眼，选择击杀目标。")
        self.memory.add_event(f"第{state['round']}夜开始")

        living_wolves = [w for w in state["werewolves"] if w in state["alive"]]
        if not living_wolves:
            state["night_victim"] = ""
            add_trace(
                state["traces"],
                round_no=state["round"],
                phase="night",
                agent="Moderator",
                thought="场上已无存活狼人",
                action="跳过夜晚击杀",
                observation="平安夜",
            )
            return state

        # Lead wolf proposes; if multiple wolves, majority / first valid.
        proposals: list[str] = []
        for wolf in living_wolves:
            proposals.append(self.agent.night_kill(state, wolf))
        proposals = [p for p in proposals if p]
        if not proposals:
            state["night_victim"] = ""
            return state

        victim = Counter(proposals).most_common(1)[0][0]
        state["night_victim"] = victim
        add_trace(
            state["traces"],
            round_no=state["round"],
            phase="night",
            agent="Moderator",
            thought=f"狼人提案统计：{dict(Counter(proposals))}",
            action=f"确认击杀目标 {victim}",
            observation="天亮后公布",
        )
        self.memory.add_event(f"第{state['round']}夜狼人选择击杀 {victim}")
        return state

    def dawn_announce(self, state: GameState) -> GameState:
        state["phase"] = "dawn"
        state["public_log"].append(f"\n## 第 {state['round']} 天 · 天亮")
        victim = state.get("night_victim") or ""
        if victim and state["players"].get(victim, {}).get("alive"):
            self._eliminate(state, victim, "昨夜遇害")
            state["public_log"].append(f"天亮了，昨夜 {victim} 被狼人杀害。")
            obs = f"公布死亡：{victim}"
        else:
            state["public_log"].append("天亮了，昨夜是平安夜，无人死亡。")
            obs = "平安夜"
            self.memory.add_event(f"第{state['round']}天平安夜")
        state["night_victim"] = ""
        add_trace(
            state["traces"],
            round_no=state["round"],
            phase="dawn",
            agent="Moderator",
            thought="根据昨夜行动公布结果",
            action="dawn_announce",
            observation=obs,
        )
        return state

    def day_speak(self, state: GameState) -> GameState:
        state["phase"] = "speak"
        state["public_log"].append(f"\n### 发言环节")
        alive = list(state["alive"])
        for name in alive:
            speech = self.agent.speak(state, name)
            record = {
                "round": state["round"],
                "speaker": name,
                "content": speech,
            }
            state["speeches"].append(record)
            self.memory.add_speech(record)
            state["public_log"].append(f"**{name}**：{speech}")
        return state

    def day_vote(self, state: GameState) -> GameState:
        state["phase"] = "vote"
        state["public_log"].append(f"\n### 投票环节")
        alive = list(state["alive"])
        ballots: list[str] = []
        for name in alive:
            target, reason = self.agent.vote(state, name)
            state["votes"].append(
                {
                    "round": state["round"],
                    "voter": name,
                    "target": target,
                    "reason": reason,
                }
            )
            ballots.append(target)
            state["public_log"].append(
                f"**{name}** 投票给 **{target}**（{reason}）"
            )

        if not ballots:
            return state

        counts = Counter(ballots)
        top = counts.most_common()
        max_votes = top[0][1]
        candidates = [name for name, c in top if c == max_votes]
        eliminated = random.choice(candidates)
        tie_note = "（平票随机处决）" if len(candidates) > 1 else ""
        state["public_log"].append(
            f"投票结果：{dict(counts)} → 处决 **{eliminated}**{tie_note}"
        )
        self._eliminate(state, eliminated, "白天投票处决")
        add_trace(
            state["traces"],
            round_no=state["round"],
            phase="vote",
            agent="Moderator",
            thought=f"计票 {dict(counts)}",
            action=f"处决 {eliminated}",
            observation="进入胜负判定",
        )
        return state

    def check_win(self, state: GameState) -> GameState:
        state["phase"] = "check"
        self._refresh_alive(state)
        living_wolves = [w for w in state["werewolves"] if w in state["alive"]]
        living_villagers = [v for v in state["villagers"] if v in state["alive"]]

        winner = _evaluate_winner(state)
        if winner == "villagers":
            msg = "【游戏结束】村民获胜：所有狼人已出局。"
        elif winner == "werewolves":
            msg = "【游戏结束】狼人获胜：存活狼人数已超过村民。"
        elif state["round"] >= state["max_rounds"]:
            # Timeout: more villagers => village win; else wolves.
            if len(living_villagers) > len(living_wolves):
                winner = "villagers"
                msg = f"【游戏结束】达到最大轮次 {state['max_rounds']}，村民以人数优势获胜。"
            else:
                winner = "werewolves"
                msg = f"【游戏结束】达到最大轮次 {state['max_rounds']}，狼人获胜。"
        else:
            msg = (
                f"胜负未分，进入第 {state['round'] + 1} 夜。"
                f"（存活狼人 {len(living_wolves)} / 村民 {len(living_villagers)}）"
            )
            state["round"] += 1
            state["public_log"].append(msg)
            self.memory.add_event(msg)
            add_trace(
                state["traces"],
                round_no=state["round"] - 1,
                phase="check",
                agent="Moderator",
                thought="双方均未达成胜利条件",
                action="continue",
                observation=msg,
            )
            return state

        state["winner"] = winner
        state["phase"] = "end"
        state["public_log"].append(msg)
        self.memory.add_event(msg)
        add_trace(
            state["traces"],
            round_no=state["round"],
            phase="check",
            agent="Moderator",
            thought="达成胜利条件或达到轮次上限",
            action=f"winner={winner}",
            observation=msg,
        )
        return state


def _evaluate_winner(state: GameState) -> str:
    """Win rules tuned for 2 wolves + 3 villagers.

    Using strict majority (wolves > villagers) avoids an instant wolf win
    after the first night kill (which would leave 2v2).
    """
    living_wolves = [w for w in state["werewolves"] if w in state["alive"]]
    living_villagers = [v for v in state["villagers"] if v in state["alive"]]
    if not living_wolves:
        return "villagers"
    if not living_villagers or len(living_wolves) > len(living_villagers):
        return "werewolves"
    return ""


def _route_after_dawn(state: GameState) -> Literal["day_speak", "check_win"]:
    """If night kill already decides the game, skip day phases."""
    if _evaluate_winner(state):
        return "check_win"
    return "day_speak"


def _route_after_check(state: GameState) -> Literal["night_action", "__end__"]:
    if state.get("winner"):
        return "__end__"
    return "night_action"


def build_graph(memory: GameMemory | None = None):
    memory = memory or GameMemory()
    nodes = ModeratorNodes(memory)

    workflow = StateGraph(GameState)
    workflow.add_node("night_action", nodes.night_action)
    workflow.add_node("dawn_announce", nodes.dawn_announce)
    workflow.add_node("day_speak", nodes.day_speak)
    workflow.add_node("day_vote", nodes.day_vote)
    workflow.add_node("check_win", nodes.check_win)

    workflow.add_edge(START, "night_action")
    workflow.add_edge("night_action", "dawn_announce")
    workflow.add_conditional_edges(
        "dawn_announce",
        _route_after_dawn,
        {
            "day_speak": "day_speak",
            "check_win": "check_win",
        },
    )
    workflow.add_edge("day_speak", "day_vote")
    workflow.add_edge("day_vote", "check_win")
    workflow.add_conditional_edges(
        "check_win",
        _route_after_check,
        {
            "night_action": "night_action",
            "__end__": END,
        },
    )
    return workflow.compile(), memory
