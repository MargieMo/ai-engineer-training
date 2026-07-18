from collections.abc import Awaitable, Callable

from .state import RetryEvent


AgentCall = Callable[[str, str], Awaitable[str]]
AskUser = Callable[[str, str], Awaitable[str]]


class AgentExecutionError(RuntimeError):
    """Raised after all three recovery levels have been exhausted."""

    def __init__(self, stage: str, events: list[RetryEvent]):
        super().__init__(f"{stage} failed after all recovery levels.")
        self.stage = stage
        self.events = events


def _detail(error: Exception) -> str:
    message = str(error).strip() or "No error message"
    return f"{type(error).__name__}: {message}"


async def execute_with_recovery(
    *,
    stage: str,
    primary_agent: str,
    backup_agent: str,
    call_agent: AgentCall,
    ask_user: AskUser,
) -> tuple[str, list[RetryEvent], str]:
    """Run one stage with same-agent retries, backup, then user input."""
    events: list[RetryEvent] = []
    last_error = "Unknown error"

    # Level 1: one initial attempt plus at most two retries.
    for attempt in range(1, 4):
        try:
            result = await call_agent(primary_agent, "")
            if attempt > 1:
                events.append(
                    {
                        "stage": stage,
                        "level": 1,
                        "attempt": attempt,
                        "agent": primary_agent,
                        "status": "recovered",
                        "detail": "The primary agent succeeded on retry.",
                    }
                )
            return result, events, ""
        except Exception as exc:  # recovery boundary intentionally catches provider/tool errors
            last_error = _detail(exc)
            events.append(
                {
                    "stage": stage,
                    "level": 1,
                    "attempt": attempt,
                    "agent": primary_agent,
                    "status": "failed",
                    "detail": last_error,
                }
            )

    # Level 2: switch role/prompt while retaining the same graph state.
    events.append(
        {
            "stage": stage,
            "level": 2,
            "attempt": 1,
            "agent": backup_agent,
            "status": "switched",
            "detail": "Primary retries exhausted; switched to the backup agent.",
        }
    )
    try:
        result = await call_agent(backup_agent, "")
        events.append(
            {
                "stage": stage,
                "level": 2,
                "attempt": 1,
                "agent": backup_agent,
                "status": "recovered",
                "detail": "The backup agent completed the stage.",
            }
        )
        return result, events, ""
    except Exception as exc:
        last_error = _detail(exc)
        events.append(
            {
                "stage": stage,
                "level": 2,
                "attempt": 1,
                "agent": backup_agent,
                "status": "failed",
                "detail": last_error,
            }
        )

    # Level 3: request more context from the user and try the backup once more.
    supplemental_context = await ask_user(stage, last_error)
    events.append(
        {
            "stage": stage,
            "level": 3,
            "attempt": 1,
            "agent": "user",
            "status": "context_received",
            "detail": "Requested supplemental information from the user.",
        }
    )
    try:
        result = await call_agent(backup_agent, supplemental_context)
        events.append(
            {
                "stage": stage,
                "level": 3,
                "attempt": 1,
                "agent": backup_agent,
                "status": "recovered",
                "detail": "The backup agent succeeded with supplemental context.",
            }
        )
        return result, events, supplemental_context
    except Exception as exc:
        events.append(
            {
                "stage": stage,
                "level": 3,
                "attempt": 1,
                "agent": backup_agent,
                "status": "failed",
                "detail": _detail(exc),
            }
        )
        raise AgentExecutionError(stage, events) from exc
