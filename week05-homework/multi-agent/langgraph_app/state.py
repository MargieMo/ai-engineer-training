from typing import NotRequired, TypedDict


class RetryEvent(TypedDict):
    """A structured record of one recovery action."""

    stage: str
    level: int
    attempt: int
    agent: str
    status: str
    detail: str


class AgentState(TypedDict):
    """Structured context shared by every LangGraph node."""

    topic: str
    style: str
    length: int
    process_log: list[str]
    retry_events: list[RetryEvent]
    user_context: str
    research_report: NotRequired[str]
    draft: NotRequired[str]
    review_suggestions: NotRequired[str]
    final_article: NotRequired[str]
