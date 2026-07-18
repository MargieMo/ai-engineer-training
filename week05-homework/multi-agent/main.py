import asyncio
import datetime as dt
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

from .langgraph_app.graph import create_graph
from .langgraph_app.retry import AgentExecutionError
from .langgraph_app.state import RetryEvent


PROJECT_DIR = Path(__file__).resolve().parent.parent


def _read_length() -> int:
    raw_value = input("请输入目标字数（直接回车使用 1000）：").strip()
    if not raw_value:
        return 1000
    try:
        value = int(raw_value)
    except ValueError:
        print("字数格式无效，使用默认值 1000。")
        return 1000
    if not 200 <= value <= 10000:
        print("字数需在 200 到 10000 之间，使用默认值 1000。")
        return 1000
    return value


def _retry_log(events: list[RetryEvent]) -> str:
    if not events:
        return "## 异常处理日志\n\n本次任务未触发重试。"

    lines = ["## 异常处理日志", ""]
    for event in events:
        lines.append(
            "- "
            f"阶段：{event['stage']} | "
            f"级别：{event['level']} | "
            f"尝试：{event['attempt']} | "
            f"执行者：{event['agent']} | "
            f"状态：{event['status']} | "
            f"详情：{event['detail']}"
        )
    return "\n".join(lines)


def _write_outputs(
    *,
    topic: str,
    final_article: str,
    process_log: list[str],
    retry_events: list[RetryEvent],
    failed: bool = False,
) -> tuple[Path | None, Path]:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    joined_log = "\n\n".join(process_log)
    process_path = PROJECT_DIR / f"process_log_{timestamp}.md"
    process_content = (
        f"# 多代理协作过程：{topic}\n\n"
        f"{joined_log}\n\n"
        "---\n\n"
        f"{_retry_log(retry_events)}\n"
    )
    process_path.write_text(process_content, encoding="utf-8")

    if failed:
        return None, process_path

    final_path = PROJECT_DIR / f"final_article_{timestamp}.md"
    final_path.write_text(
        f"# {topic}\n\n{final_article}\n",
        encoding="utf-8",
    )
    return final_path, process_path


async def run_writing_task() -> None:
    """Connect to MCP and execute the complete writing workflow."""
    load_dotenv(PROJECT_DIR / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "缺少 OPENAI_API_KEY。请复制 .env.example 为 .env 并填写密钥。"
        )

    mcp_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")
    client = MultiServerMCPClient(
        {
            "writer_tools": {
                "url": mcp_url,
                "transport": "streamable_http",
            }
        }
    )

    async with client.session("writer_tools") as mcp_session:
        print(f"✅ 已连接 MCP 服务：{mcp_url}")
        graph = await create_graph(mcp_session)

        topic = input("请输入文章主题（直接回车使用默认主题）：").strip()
        topic = topic or "帮我写一篇关于 AI Agent 的文章"
        style = input("请输入文章风格（直接回车使用“通俗易懂”）：").strip()
        style = style or "通俗易懂"
        length = _read_length()

        initial_state: dict[str, Any] = {
            "topic": topic,
            "style": style,
            "length": length,
            "user_context": "",
            "process_log": [
                (
                    "## 任务配置\n\n"
                    f"- 主题：{topic}\n"
                    f"- 风格：{style}\n"
                    f"- 目标字数：{length}\n"
                    f"- 模型：{os.getenv('OPENAI_MODEL', 'gpt-5')}"
                )
            ],
            "retry_events": [],
        }

        print("\n🚀 开始执行：研究 → 撰写 → 审核 → 润色\n")
        try:
            final_state = await graph.ainvoke(initial_state)
        except AgentExecutionError as exc:
            _, process_path = _write_outputs(
                topic=topic,
                final_article=f"{exc.stage}阶段在三级恢复后仍然失败。",
                process_log=initial_state["process_log"],
                retry_events=exc.events,
                failed=True,
            )
            print(f"❌ 任务失败，过程记录已写入：{process_path}")
            raise

        final_path, process_path = _write_outputs(
            topic=topic,
            final_article=final_state["final_article"],
            process_log=final_state["process_log"],
            retry_events=final_state["retry_events"],
        )
        print(f"\n🎉 任务完成")
        print(f"  终稿：{final_path}")
        print(f"  过程记录：{process_path}")


def main() -> None:
    """CLI entry point. Run with: python -m multi-agent.main"""
    try:
        asyncio.run(run_writing_task())
    except KeyboardInterrupt:
        print("\n程序已由用户中断。")
    except AgentExecutionError as exc:
        print(f"\n{exc}")
    except Exception as exc:
        print(f"\n运行失败：{exc}")
        print("请确认 MCP 服务已启动，且 OpenAI 与搜索网络连接可用。")


if __name__ == "__main__":
    main()