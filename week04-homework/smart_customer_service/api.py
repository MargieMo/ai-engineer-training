from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from .services import service_manager
from .graph import GraphManager


app = FastAPI(
    title="Smart Customer Service API",
    description="A smart customer service API built with LangGraph and hot-reloadable tools/models.",
    version="1.0.0",
)


# GraphManager depends on the shared ServiceManager singleton.
graph_manager = GraphManager(service_manager)


class ChatRequest(BaseModel):
    user_id: str  # Used to track a conversation (mapped to a LangGraph thread).
    query: str


class HotUpdateRequest(BaseModel):
    type: str  # "model" or "tools"
    name: str  # e.g. "gpt-5" or "query_only" / "default"


@app.get("/health", summary="Health check")
async def health_check():
    """Report whether the service is healthy and what model / tools are active."""
    return {"status": "healthy", "services": service_manager.get_services_status()}


@app.post("/chat", summary="Chat with the assistant")
async def chat(request: ChatRequest):
    """Send one turn to the assistant. Conversation memory is keyed by user_id."""
    thread_id = request.user_id
    config = {"configurable": {"thread_id": thread_id}}

    messages = [HumanMessage(content=request.query)]

    # Always use the current graph instance (it may have been hot-reloaded).
    current_app = graph_manager.get_app()

    final_response = ""
    # Stream through the graph and keep the last non-tool AI message as the reply.
    for event in current_app.stream({"messages": messages}, config=config, stream_mode="values"):
        if "messages" in event:
            last_message = event["messages"][-1]
            if isinstance(last_message, AIMessage) and not last_message.tool_calls:
                final_response = last_message.content

    if not final_response:
        return {"user_id": thread_id, "response": "Sorry, I can't answer that right now."}

    return {"user_id": thread_id, "response": final_response}


@app.post("/hot-update", summary="Hot update the model or tools")
async def hot_update(request: HotUpdateRequest):
    """Perform a hot update of the model or tools without restarting the service.

    - type: 'model', name: 'gpt-5'
    - type: 'tools', name: 'query_only' (or anything else to restore defaults)
    """
    try:
        if request.type == "model":
            service_manager.update_llm(request.name)
        elif request.type == "tools":
            if request.name == "query_only":
                from .tools.order_tools import query_order
                service_manager.update_tools([query_order])
            else:  # Restore the default tool set.
                from .tools import default_tools
                service_manager.update_tools(default_tools)
        else:
            raise HTTPException(status_code=400, detail="Invalid update type.")

        # Rebuild the graph so new sessions use the updated model / tools.
        graph_manager.reload_graph()

        return {"status": "success", "message": f"{request.type} hot update completed."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hot update failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
