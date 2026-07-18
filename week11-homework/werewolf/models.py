"""Domain models and LangGraph state for the werewolf game."""

from __future__ import annotations

from enum import Enum
from typing import Any, TypedDict


class Role(str, Enum):
    WEREWOLF = "werewolf"
    VILLAGER = "villager"


class Personality(str, Enum):
    AGGRESSIVE = "aggressive"
    CAUTIOUS = "cautious"
    ANALYTICAL = "analytical"


class Phase(str, Enum):
    NIGHT = "night"
    DAWN = "dawn"
    SPEAK = "speak"
    VOTE = "vote"
    CHECK = "check"
    END = "end"


class Player:
    def __init__(
        self,
        name: str,
        role: Role,
        personality: Personality,
    ) -> None:
        self.name = name
        self.role = role
        self.personality = personality
        self.alive = True

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "alive": self.alive,
            "personality": self.personality.value,
        }


class TraceRecord(TypedDict):
    round: int
    phase: str
    agent: str
    thought: str
    action: str
    observation: str


class SpeechRecord(TypedDict):
    round: int
    speaker: str
    content: str


class VoteRecord(TypedDict):
    round: int
    voter: str
    target: str
    reason: str


class CostStats(TypedDict):
    calls: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    total_latency_ms: float
    latencies_ms: list[float]


class GameState(TypedDict):
    """LangGraph shared state."""

    round: int
    phase: str
    players: dict[str, dict[str, Any]]
    alive: list[str]
    werewolves: list[str]
    villagers: list[str]
    night_victim: str
    speeches: list[SpeechRecord]
    votes: list[VoteRecord]
    eliminated: list[str]
    winner: str
    traces: list[TraceRecord]
    public_log: list[str]
    cost: CostStats
    max_rounds: int
