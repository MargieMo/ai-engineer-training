"""Phase 1 basic chain: Prompt -> LLM -> OutputParser.

Demonstrates resolving relative time expressions (e.g. "我昨天下的单") into
concrete dates using the current date as context. Stage 2+ routes the same
problem through LangGraph + tools; this module keeps the README stage-one
pattern explicit and runnable on its own.
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .date_context import today_str

RELATIVE_TIME_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a date parsing assistant for an e-commerce customer service bot. "
        "Today is {today} (YYYY-MM-DD).\n\n"
        "Read the user's message and resolve any relative time expressions "
        "(e.g. 昨天/yesterday, 前天, 上周三/last Wednesday) into concrete dates.\n"
        "Respond in one short sentence that states the resolved date(s) in YYYY-MM-DD format. "
        "If the message mentions placing an order on a relative day, explain which calendar "
        "date that refers to. If no relative time is present, say so briefly.",
    ),
    ("human", "{user_input}"),
])


def build_relative_time_chain(llm: ChatOpenAI):
    """Build the stage-one LCEL chain: Prompt -> LLM -> OutputParser."""
    return RELATIVE_TIME_PROMPT | llm | StrOutputParser()


def resolve_relative_time(user_input: str, llm: ChatOpenAI | None = None) -> str:
    """Run the relative-time chain for a single user utterance."""
    if llm is None:
        from .services import service_manager
        llm = service_manager.get_llm()

    chain = build_relative_time_chain(llm)
    return chain.invoke({"today": today_str(), "user_input": user_input})


def run_phase1_demo() -> None:
    """Interactive demo for README stage one."""
    from .services import service_manager

    llm = service_manager.get_llm()
    chain = build_relative_time_chain(llm)

    samples = [
        "我昨天下的单",
        "I placed an order last Wednesday",
        "查一下我前天的退款进度",
    ]

    print("\n=== Phase 1: Prompt -> LLM -> OutputParser ===")
    current_today = today_str()
    print(f"Today (system date): {current_today}\n")

    for sample in samples:
        print(f"User: {sample}")
        answer = chain.invoke({"today": current_today, "user_input": sample})
        print(f"Chain: {answer}\n")

    print("Type your own messages (empty line to exit):\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nDone.")
            break
        if not user_input:
            break
        answer = chain.invoke({"today": today_str(), "user_input": user_input})
        print(f"Chain: {answer}\n")
