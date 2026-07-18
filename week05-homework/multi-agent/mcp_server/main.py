import os
from typing import Any

from ddgs import DDGS
from dotenv import load_dotenv
from fastmcp import FastMCP

from .prompts import PROMPTS


load_dotenv()
mcp = FastMCP("Multi-Agent Writer Tools")


@mcp.tool
def search(topic: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Search the web with DuckDuckGo and return structured results."""
    limit = max(1, min(max_results, 10))
    print(f"MCP Server: searching for {topic!r}")
    results = list(
        DDGS().text(
            topic,
            region="us-en",
            safesearch="moderate",
            max_results=limit,
        )
    )
    if not results:
        raise RuntimeError(f"No search results were returned for {topic!r}.")
    return results


@mcp.tool
def get_prompt(agent_name: str) -> str:
    """Return the system prompt for a primary or backup agent."""
    try:
        return PROMPTS[agent_name]
    except KeyError as exc:
        raise ValueError(f"Unknown agent name: {agent_name}") from exc


@mcp.tool
def health_check() -> dict[str, str]:
    """Return MCP server health information."""
    return {"status": "ok", "service": "multi-agent-writer"}


def run() -> None:
    """Run the MCP server over streamable HTTP."""
    host = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_SERVER_PORT", "8000"))
    print(f"MCP Server running at http://{host}:{port}/mcp")
    mcp.run(transport="streamable-http", host=host, port=port)


if __name__ == "__main__":
    run()
