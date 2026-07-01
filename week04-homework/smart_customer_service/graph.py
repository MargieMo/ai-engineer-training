import operator
from typing import Annotated, Sequence, Literal, TypedDict
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from .services import ServiceManager


def build_system_prompt(tools: list) -> str:
    """Build a system prompt that only advertises capabilities present in *tools*."""
    tool_names = {tool.name for tool in tools}
    capabilities = []
    if "query_order" in tool_names:
        capabilities.append("query orders")
    if "apply_refund" in tool_names:
        capabilities.append("apply for refunds")
    if "generate_invoice" in tool_names:
        capabilities.append("generate invoices")
    if "get_date_for_relative_time" in tool_names:
        capabilities.append("resolve relative dates such as 'yesterday'")

    if capabilities:
        capability_text = ", ".join(capabilities)
    else:
        capability_text = "answer general questions (no backend tools are currently available)"

    unavailable = []
    if "apply_refund" not in tool_names:
        unavailable.append("refunds")
    if "generate_invoice" not in tool_names:
        unavailable.append("invoices")
    if "query_order" not in tool_names:
        unavailable.append("order lookups")

    prompt = (
        "You are a helpful e-commerce customer service assistant. "
        f"Your available capabilities are: {capability_text}. "
        "When a user refers to a relative date, call the date tool first if it is available. "
        "Always ask for the order id if it is required but missing. "
        "Be concise and friendly. "
        "Only call tools that are currently bound to you; never claim an action was completed "
        "unless a tool returned a successful result."
    )
    if unavailable:
        prompt += (
            " The following services are temporarily unavailable: "
            + ", ".join(unavailable)
            + ". If the user asks for them, politely explain you cannot perform that action "
            "right now and offer help with the capabilities you still have."
        )
    return prompt


class AgentState(TypedDict):
    # `operator.add` makes new messages append to the running history.
    messages: Annotated[Sequence[BaseMessage], operator.add]


class GraphManager:
    def __init__(self, service_manager: ServiceManager):
        self.service_manager = service_manager
        # A single checkpointer keeps per-thread conversation memory alive
        # across graph rebuilds, so hot updates do not wipe existing sessions.
        self.checkpointer = MemorySaver()
        self._app = self._build_graph()

    def _build_graph(self):
        """Build (or rebuild) the LangGraph application."""
        workflow = StateGraph(AgentState)

        tools = self.service_manager.get_tools()
        tool_node = ToolNode(tools)

        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", tool_node)
        workflow.add_node("ask_for_order_id", self._ask_for_order_id)

        workflow.set_conditional_entry_point(
            self._router,
            {
                "ask_for_order_id": "ask_for_order_id",
                "agent": "agent",
            },
        )
        workflow.add_edge("ask_for_order_id", END)
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {"tools": "tools", "end": END},
        )
        workflow.add_edge("tools", "agent")

        app = workflow.compile(checkpointer=self.checkpointer)
        print("LangGraph graph built/rebuilt successfully!")
        return app

    def get_app(self):
        """Return the compiled LangGraph application instance."""
        return self._app

    def reload_graph(self) -> None:
        """Hot-reload the graph so it picks up the latest model / tools."""
        self._app = self._build_graph()

    def _call_model(self, state: AgentState):
        print("--- [Node] Agent: Thinking... ---")
        try:
            llm = self.service_manager.get_llm()
            tools = self.service_manager.get_tools()
            model_with_tools = llm.bind_tools(tools)
            system_prompt = build_system_prompt(tools)
            messages = [SystemMessage(content=system_prompt), *state["messages"]]
            response = model_with_tools.invoke(messages)
            return {"messages": [response]}
        except Exception as e:
            print(f"Model invocation error: {e}")
            return {"messages": [AIMessage(content="Sorry, the system ran into an error. Please try again later.")]}

    @staticmethod
    def _router(state: AgentState) -> Literal["agent", "ask_for_order_id"]:
        print("--- [Node] Router: Analyzing user intent... ---")
        last_message = state["messages"][-1]
        content = last_message.content or ""
        # Simple keyword-based routing; could be replaced with an intent model later.
        wants_order = "查订单" in content or "check order" in content.lower()
        if wants_order and "SN" not in content:
            # If the user mentions a relative date, let the agent handle it (tools).
            if any(kw in content for kw in ["昨天", "前天", "今天", "上周", "yesterday", "today", "last week"]):
                return "agent"
            print("--- [Decision] Routing to 'ask_for_order_id'. ---")
            return "ask_for_order_id"
        print("--- [Decision] Routing to 'agent'. ---")
        return "agent"

    @staticmethod
    def _ask_for_order_id(state: AgentState):
        print("--- [Node] ask_for_order_id: Generating a follow-up question. ---")
        follow_up_message = AIMessage(content="Sure, could you tell me your order id?")
        return {"messages": [follow_up_message]}

    @staticmethod
    def _should_continue(state: AgentState) -> Literal["tools", "end"]:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            print("--- [Decision] LLM requested a tool call, routing to tool node. ---")
            return "tools"
        print("--- [Decision] No tool call, ending turn. ---")
        return "end"
