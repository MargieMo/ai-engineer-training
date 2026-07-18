import importlib
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
retry_module = importlib.import_module("multi-agent.langgraph_app.retry")
length_module = importlib.import_module("multi-agent.langgraph_app.length")
AgentExecutionError = retry_module.AgentExecutionError
execute_with_recovery = retry_module.execute_with_recovery
length_bounds = length_module.length_bounds
validate_length = length_module.validate_length


@pytest.mark.asyncio
async def test_primary_agent_retries_twice_then_recovers():
    calls = []

    async def call_agent(agent_name: str, context: str) -> str:
        calls.append((agent_name, context))
        if len(calls) < 3:
            raise RuntimeError("temporary failure")
        return "success"

    async def ask_user(stage: str, error: str) -> str:
        raise AssertionError("level 3 should not run")

    result, events, context = await execute_with_recovery(
        stage="研究",
        primary_agent="research",
        backup_agent="research_backup",
        call_agent=call_agent,
        ask_user=ask_user,
    )

    assert result == "success"
    assert context == ""
    assert [name for name, _ in calls] == ["research", "research", "research"]
    assert events[-1]["status"] == "recovered"
    assert events[-1]["attempt"] == 3


@pytest.mark.asyncio
async def test_switches_to_backup_after_primary_retries():
    calls = []

    async def call_agent(agent_name: str, context: str) -> str:
        calls.append((agent_name, context))
        if agent_name == "primary":
            raise RuntimeError("primary unavailable")
        return "backup success"

    async def ask_user(stage: str, error: str) -> str:
        raise AssertionError("level 3 should not run")

    result, events, _ = await execute_with_recovery(
        stage="审核",
        primary_agent="primary",
        backup_agent="backup",
        call_agent=call_agent,
        ask_user=ask_user,
    )

    assert result == "backup success"
    assert [name for name, _ in calls] == [
        "primary",
        "primary",
        "primary",
        "backup",
    ]
    assert any(event["level"] == 2 for event in events)


@pytest.mark.asyncio
async def test_requests_user_context_before_final_attempt():
    calls = []

    async def call_agent(agent_name: str, context: str) -> str:
        calls.append((agent_name, context))
        if context == "more details":
            return "recovered with context"
        raise RuntimeError("needs context")

    async def ask_user(stage: str, error: str) -> str:
        return "more details"

    result, events, context = await execute_with_recovery(
        stage="润色",
        primary_agent="primary",
        backup_agent="backup",
        call_agent=call_agent,
        ask_user=ask_user,
    )

    assert result == "recovered with context"
    assert context == "more details"
    assert calls[-1] == ("backup", "more details")
    assert events[-1]["level"] == 3
    assert events[-1]["status"] == "recovered"


@pytest.mark.asyncio
async def test_raises_after_all_recovery_levels_fail():
    async def call_agent(agent_name: str, context: str) -> str:
        raise RuntimeError("permanent failure")

    async def ask_user(stage: str, error: str) -> str:
        return "extra context"

    with pytest.raises(AgentExecutionError) as caught:
        await execute_with_recovery(
            stage="撰写",
            primary_agent="primary",
            backup_agent="backup",
            call_agent=call_agent,
            ask_user=ask_user,
        )

    assert caught.value.events[-1]["level"] == 3
    assert caught.value.events[-1]["status"] == "failed"


def test_length_bounds_for_target_500():
    assert length_bounds(500) == (425, 575)


def test_validate_length_rejects_too_long_text():
    with pytest.raises(ValueError, match="过长"):
        validate_length("a" * 600, 500, stage="润色")


def test_validate_length_accepts_text_within_bounds():
    validate_length("a" * 500, 500, stage="润色")
