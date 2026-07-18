"""Player agent decisions: speak / vote / night kill."""

from __future__ import annotations

import json
import random
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .memory import GameMemory
from .models import GameState
from .prompts import (
    NIGHT_KILL_INSTRUCTION,
    SPEAK_INSTRUCTION,
    VOTE_INSTRUCTION,
    build_system_prompt,
)
from .tracing import add_trace, invoke_llm


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def _alive_names(state: GameState) -> list[str]:
    return list(state["alive"])


def _player_meta(state: GameState, name: str) -> dict[str, Any]:
    return state["players"][name]


class PlayerAgent:
    def __init__(self, llm: ChatOpenAI, memory: GameMemory) -> None:
        self.llm = llm
        self.memory = memory

    def _system(self, state: GameState, name: str) -> str:
        meta = _player_meta(state, name)
        teammates = state["werewolves"] if meta["role"] == "werewolf" else None
        return build_system_prompt(
            name=name,
            role=meta["role"],
            personality=meta["personality"],
            teammates=teammates,
        )

    def speak(self, state: GameState, name: str) -> str:
        alive = _alive_names(state)
        today = [
            s for s in state["speeches"] if s["round"] == state["round"]
        ]
        already = (
            "\n".join(f"- {s['speaker']}: {s['content']}" for s in today)
            if today
            else "（你是本轮第一位发言者）"
        )
        query = (
            f"第{state['round']}天发言，当前存活{alive}，"
            f"我是{name}，需要参考谁可能是狼人、谁的发言有矛盾"
        )
        rag = self.memory.retrieve(query)
        human = SPEAK_INSTRUCTION.format(
            round=state["round"],
            alive="、".join(alive),
            eliminated="、".join(state["eliminated"]) or "无",
            already_spoken=already,
            situation=self.memory.situation_summary(),
            rag_context=rag,
        )
        raw = invoke_llm(
            self.llm,
            [
                SystemMessage(content=self._system(state, name)),
                HumanMessage(content=human),
            ],
            state["cost"],
        )
        data = _parse_json(raw)
        thought = str(data.get("thought") or "（模型未返回 thought）")
        speech = str(data.get("speech") or raw.strip()[:200])
        add_trace(
            state["traces"],
            round_no=state["round"],
            phase="speak",
            agent=name,
            thought=thought,
            action=f"发言: {speech}",
            observation="发言已记录并写入语义记忆",
        )
        return speech

    def vote(self, state: GameState, name: str) -> tuple[str, str]:
        alive = [n for n in _alive_names(state) if n != name]
        today = self.memory.today_speeches(state["speeches"], state["round"])
        query = f"第{state['round']}天投票，怀疑谁是狼人？今日发言：{today}"
        rag = self.memory.retrieve(query)
        human = VOTE_INSTRUCTION.format(
            round=state["round"],
            alive="、".join(alive),
            today_speeches=today,
            rag_context=rag,
        )
        raw = invoke_llm(
            self.llm,
            [
                SystemMessage(content=self._system(state, name)),
                HumanMessage(content=human),
            ],
            state["cost"],
        )
        data = _parse_json(raw)
        thought = str(data.get("thought") or "（模型未返回 thought）")
        target = str(data.get("target") or "")
        reason = str(data.get("reason") or "综合发言风险投票")
        if target not in alive:
            target = random.choice(alive)
            observation = f"目标非法，主持人纠正为随机存活玩家 {target}"
        else:
            observation = f"投票生效：{name} → {target}"
        add_trace(
            state["traces"],
            round_no=state["round"],
            phase="vote",
            agent=name,
            thought=thought,
            action=f"投票处决 {target}；理由：{reason}",
            observation=observation,
        )
        return target, reason

    def night_kill(self, state: GameState, name: str) -> str:
        villagers = [
            n for n in state["alive"] if n in state["villagers"]
        ]
        if not villagers:
            return ""
        teammates = [w for w in state["werewolves"] if w in state["alive"]]
        human = NIGHT_KILL_INSTRUCTION.format(
            round=state["round"],
            villager_targets="、".join(villagers),
            teammates="、".join(teammates),
            situation=self.memory.situation_summary(),
        )
        raw = invoke_llm(
            self.llm,
            [
                SystemMessage(content=self._system(state, name)),
                HumanMessage(content=human),
            ],
            state["cost"],
        )
        data = _parse_json(raw)
        thought = str(data.get("thought") or "（模型未返回 thought）")
        target = str(data.get("target") or "")
        if target not in villagers:
            target = random.choice(villagers)
            observation = f"击杀目标非法，主持人纠正为 {target}"
        else:
            observation = f"狼人行动登记：意图击杀 {target}"
        add_trace(
            state["traces"],
            round_no=state["round"],
            phase="night",
            agent=name,
            thought=thought,
            action=f"夜晚击杀 {target}",
            observation=observation,
        )
        return target
