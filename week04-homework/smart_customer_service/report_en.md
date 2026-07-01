# Smart Customer Service - Experiment Report (English)

> For the Chinese version, see [`report.md`](./report.md).

## 1. Overview

This project implements a smart customer service system built on LangChain and LangGraph, powered by **OpenAI** (default `gpt-5`, integrated through `ChatOpenAI` from `langchain-openai`). It supports multi-turn dialogue management, automatic tool calling, and hot updates of both the model and the plugins (tools). It can handle order lookups, refund requests, invoice generation, and relative-date resolution, and it can swap the model or tools at runtime without restarting the service.

## 2. Architecture

The system is split into five layers by responsibility:

- **Stage-one chain `base_chain.py` (Prompt -> LLM -> OutputParser)**: A standalone LCEL chain that resolves relative dates (e.g. "我昨天下的单") using a fixed anchor date; run via `python -m smart_customer_service.main phase1`.
- **Service layer `services.py` (`ServiceManager` singleton)**: The single source of truth that owns the LLM and the tool set, exposing `update_llm()` and `update_tools()` for hot updates.
- **Orchestration layer `graph.py` (`GraphManager` + LangGraph state graph)**: Orchestrates the dialogue flow and uses a `MemorySaver` checkpointer to keep per-thread (per-session) conversation memory.
- **API layer `api.py` (FastAPI)**: Exposes three endpoints: `/health`, `/chat`, and `/hot-update`.
- **Entry point `main.py`**: `server` starts the API, `phase1` runs the stage-one chain demo, `cli` provides LangGraph console chat.

## 3. Core Features and Test Results

### 3.1 Stage One: Basic Chain (Relative Time Inference)

`base_chain.py` explicitly implements **Prompt -> LLM -> OutputParser** with LangChain LCEL:

- **Prompt**: Injects the anchor date (2025-09-26) and the user message into the system prompt.
- **LLM**: The OpenAI model resolves expressions such as "yesterday" or "last Wednesday".
- **OutputParser**: `StrOutputParser` returns the resolved date in natural language.

Run `python -m smart_customer_service.main phase1` to try built-in samples and interact. Stage two and beyond still handle relative dates through LangGraph plus the `get_date_for_relative_time` tool.

### 3.2 Dialogue System

A state-driven dialogue flow is built with LangGraph, containing these nodes:

- **agent node**: Calls the OpenAI model (with tools bound) and decides whether to trigger a tool call.
- **tools node**: Executes the concrete tools, such as querying orders, applying refunds, generating invoices, and parsing dates.
- **ask_for_order_id node**: When the user asks to check an order without providing an order id, it proactively asks for the id.

Tests confirm correct intent handling: a missing order id triggers a follow-up question, while a relative date such as "yesterday" / "last Wednesday" is routed to the agent so it can call the date tool first. This demonstrates solid multi-turn dialogue management.

### 3.3 Tool Calling

Four tools are implemented:

1. **query_order**: Looks up order status and shipping info by order id (backed by mock data).
2. **apply_refund**: Applies for a refund for a given order and returns a refund id.
3. **generate_invoice**: Generates an invoice download link for a given order.
4. **get_date_for_relative_time**: Converts relative dates ("yesterday", "last Wednesday", and their Chinese equivalents) into a concrete `YYYY-MM-DD` date.

Automated tests verify that the invoice tool produces a URL containing the order id, that an invalid order id fails gracefully, and that relative-time parsing is correct (relative to 2025-09-26, "yesterday" = 2025-09-25).

### 3.4 Hot Updates

Two kinds of hot updates are supported:

1. **Model hot update**: Switch the OpenAI model at runtime (e.g. `gpt-5` -> `gpt-4o`).
2. **Tool hot update**: Replace the available tool set at runtime (e.g. keep only the query tool).

After an update, `reload_graph()` rebuilds the graph instance so new sessions use the new configuration, while in-flight sessions can finish their lifecycle. Combined with LangGraph's `thread_id` and `MemorySaver`, this isolates session state and preserves service continuity.

## 4. Quick Start

```bash
cd week04-homework
uv sync                       # install dependencies
cp .env.example .env          # set OPENAI_API_KEY

# Option 1: stage-one chain demo
python -m smart_customer_service.main phase1

# Option 2: start the API server
python -m smart_customer_service.main server
# Health check
curl http://localhost:8000/health
# Chat
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"user_id":"u1","query":"I want to check my order"}'
# Hot update the model
curl -X POST http://localhost:8000/hot-update -H "Content-Type: application/json" \
  -d '{"type":"model","name":"gpt-5"}'

# Option 3: interactive console mode
python -m smart_customer_service.main cli

# Run the tests
python -m unittest tests.test_features -v
```

## 5. Performance and Stability

- **Latency**: Dominated by the LLM call (~1-3s) and simulated tool latency (~1-2s); the framework overhead itself is negligible.
- **Stability**: Model calls are wrapped in exception handling to avoid crashes; the `/health` endpoint enables monitoring; session isolation is achieved via `thread_id` plus the checkpointer.

## 6. Known Limitations and Future Work

- **Simple intent detection**: The entry router relies on keyword matching and could be replaced by a dedicated intent model.
- **Mock tools**: Order/refund/invoice data are simulated; a real deployment would integrate a real backend and database.
- **In-memory memory**: `MemorySaver` is lost on restart; production should use a persistent checkpointer (e.g. a database).
- **Error handling**: More retry / fallback strategies could be added for complex failure scenarios.

## 7. Conclusion

The project delivers a complete, OpenAI-backed smart customer service system covering all three homework stages: basic dialogue with relative-time inference, LangGraph-based multi-turn dialogue with tool calling, and hot updates / health checks / automated tests. The design is clean and extensible, and the hot-update mechanism gives it the maintainability needed for production.
