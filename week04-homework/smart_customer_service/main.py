"""Entry point for the smart customer service homework.

Usage (run from the week04-homework root):
    uv sync
    uv run python -m smart_customer_service.main            # start the FastAPI server
    uv run python -m smart_customer_service.main server     # same as above
    uv run python -m smart_customer_service.main phase1     # stage-one Prompt->LLM->OutputParser demo
    uv run python -m smart_customer_service.main cli        # interactive LangGraph console chat

If you use plain `python`, activate the project venv first:
    source .venv/bin/activate
"""
import sys
import uuid


def _exit_missing_deps(package: str) -> None:
    print(f"Error: missing package '{package}'.")
    print("Install project dependencies from week04-homework, then rerun with:")
    print("  uv sync")
    print("  uv run python -m smart_customer_service.main server")
    print("Or activate the venv: source .venv/bin/activate")
    sys.exit(1)


def run_server() -> None:
    """Start the FastAPI app with uvicorn (exposes /health, /chat, /hot-update)."""
    try:
        import uvicorn
    except ModuleNotFoundError:
        _exit_missing_deps("uvicorn")
    print("Starting Smart Customer Service API on http://0.0.0.0:8000 ...")
    print("Docs available at http://0.0.0.0:8000/docs")
    uvicorn.run("smart_customer_service.api:app", host="0.0.0.0", port=8000, reload=False)


def run_phase1() -> None:
    """Run the README stage-one chain demo (relative time inference)."""
    from .base_chain import run_phase1_demo
    run_phase1_demo()


def run_cli() -> None:
    """Run an interactive multi-turn chat loop in the terminal."""
    from langchain_core.messages import HumanMessage, AIMessage
    from .services import service_manager
    from .graph import GraphManager

    graph_manager = GraphManager(service_manager)
    app = graph_manager.get_app()

    # One thread id per CLI session keeps the conversation memory together.
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("\nSmart Customer Service (CLI mode). Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not query:
            continue

        final_response = ""
        for event in app.stream(
            {"messages": [HumanMessage(content=query)]},
            config=config,
            stream_mode="values",
        ):
            last_message = event["messages"][-1]
            if isinstance(last_message, AIMessage) and not last_message.tool_calls:
                final_response = last_message.content

        print(f"Assistant: {final_response or 'Sorry, I could not process that.'}\n")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "server"
    if mode == "phase1":
        run_phase1()
    elif mode == "cli":
        run_cli()
    else:
        run_server()


if __name__ == "__main__":
    main()
