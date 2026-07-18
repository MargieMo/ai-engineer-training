import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .length import length_bounds, validate_length
from .retry import AskUser, execute_with_recovery
from .state import AgentState


InputBuilder = Callable[[str], Awaitable[str]]


class AgentNodes:
    """Four specialist agents backed by GPT-5 and MCP-provided tools."""

    def __init__(
        self,
        mcp_tools: list[Any],
        *,
        llm: Any | None = None,
        ask_user: AskUser | None = None,
    ):
        self.mcp_tools = {tool.name: tool for tool in mcp_tools}
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-5")
        self.llm = llm or ChatOpenAI(model=self.model_name)
        self.ask_user = ask_user or self._ask_user

    async def _call_mcp_tool(self, tool_name: str, **kwargs: Any) -> Any:
        try:
            tool = self.mcp_tools[tool_name]
        except KeyError as exc:
            raise RuntimeError(f"MCP tool {tool_name!r} is unavailable.") from exc
        return await tool.ainvoke(kwargs)

    @staticmethod
    def _as_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            text_blocks = [
                str(block.get("text", ""))
                for block in value
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            if text_blocks:
                return "\n".join(text_blocks)
        return json.dumps(value, ensure_ascii=False, indent=2)

    @staticmethod
    def _response_text(response: Any) -> str:
        content = response.content
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            return "\n".join(parts).strip()
        return str(content).strip()

    @staticmethod
    async def _ask_user(stage: str, error: str) -> str:
        prompt = (
            f"\n三级恢复已启动：{stage}阶段仍然失败。\n"
            f"最后错误：{error}\n"
            "请补充有助于继续任务的信息（直接回车表示无补充）："
        )
        try:
            answer = await asyncio.to_thread(input, prompt)
        except EOFError:
            answer = ""
        return answer.strip() or "用户没有提供额外信息，请基于现有上下文完成任务。"

    async def _run_stage(
        self,
        state: AgentState,
        *,
        stage: str,
        primary_agent: str,
        backup_agent: str,
        output_key: str,
        log_title: str,
        build_input: InputBuilder,
        min_chars: int,
        max_chars: int | None = None,
        include_body_in_log: bool = True,
    ) -> dict[str, Any]:
        print(f"--- {stage}代理（模型：{self.model_name}）---")

        async def call_agent(agent_name: str, supplemental_context: str) -> str:
            prompt_value = await self._call_mcp_tool(
                "get_prompt", agent_name=agent_name
            )
            system_prompt = self._as_text(prompt_value).format(
                style=state["style"],
                length=state["length"],
            )
            user_input = await build_input(supplemental_context)
            response = await self.llm.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_input),
                ]
            )
            result = self._response_text(response)
            if max_chars is None:
                if len(result) < min_chars:
                    raise ValueError(
                        f"{stage}代理返回内容过短（{len(result)} 字符，"
                        f"至少需要 {min_chars}）。"
                    )
            else:
                validate_length(
                    result,
                    state["length"],
                    stage=stage,
                )
            return result

        result, retry_events, supplemental_context = await execute_with_recovery(
            stage=stage,
            primary_agent=primary_agent,
            backup_agent=backup_agent,
            call_agent=call_agent,
            ask_user=self.ask_user,
        )

        if include_body_in_log:
            log_entry = f"## {log_title}\n\n{result}"
        else:
            log_entry = (
                f"## {log_title}\n\n"
                f"终稿已完成，字符数 {len(result)}（目标 {state['length']}）。"
                "正文已写入独立的 final_article 文件，此处不再重复粘贴。"
            )

        update: dict[str, Any] = {
            output_key: result,
            "process_log": state["process_log"] + [log_entry],
            "retry_events": state["retry_events"] + retry_events,
        }
        if supplemental_context:
            previous = state.get("user_context", "").strip()
            update["user_context"] = "\n".join(
                item for item in (previous, supplemental_context) if item
            )

        print(f"✅ {stage}阶段完成。")
        return update

    async def research_node(self, state: AgentState) -> dict[str, Any]:
        async def build_input(supplemental_context: str) -> str:
            max_results = int(os.getenv("SEARCH_MAX_RESULTS", "5"))
            search_results = await self._call_mcp_tool(
                "search",
                topic=state["topic"],
                max_results=max_results,
            )
            return (
                f"主题：{state['topic']}\n\n"
                f"已有用户补充信息：{state.get('user_context') or '无'}\n\n"
                f"本次补充信息：{supplemental_context or '无'}\n\n"
                f"搜索结果：\n{self._as_text(search_results)}"
            )

        return await self._run_stage(
            state,
            stage="研究",
            primary_agent="research",
            backup_agent="research_backup",
            output_key="research_report",
            log_title="研究报告",
            build_input=build_input,
            min_chars=120,
        )

    async def writing_node(self, state: AgentState) -> dict[str, Any]:
        async def build_input(supplemental_context: str) -> str:
            return (
                f"文章主题：{state['topic']}\n\n"
                f"研究报告：\n{state['research_report']}\n\n"
                f"用户补充信息：{state.get('user_context') or '无'}\n"
                f"{supplemental_context}"
            )

        return await self._run_stage(
            state,
            stage="撰写",
            primary_agent="write",
            backup_agent="write_backup",
            output_key="draft",
            log_title="文章初稿",
            build_input=build_input,
            min_chars=200,
        )

    async def review_node(self, state: AgentState) -> dict[str, Any]:
        async def build_input(supplemental_context: str) -> str:
            return (
                f"文章主题：{state['topic']}\n\n"
                f"研究报告：\n{state['research_report']}\n\n"
                f"文章初稿：\n{state['draft']}\n\n"
                f"本次补充信息：{supplemental_context or '无'}"
            )

        return await self._run_stage(
            state,
            stage="审核",
            primary_agent="review",
            backup_agent="review_backup",
            output_key="review_suggestions",
            log_title="审核建议",
            build_input=build_input,
            min_chars=20,
        )

    async def polishing_node(self, state: AgentState) -> dict[str, Any]:
        min_chars, max_chars = length_bounds(state["length"])

        async def build_input(supplemental_context: str) -> str:
            return (
                f"文章主题：{state['topic']}\n\n"
                f"目标字数：{state['length']} 个中文字\n"
                f"研究报告：\n{state['research_report']}\n\n"
                f"文章初稿：\n{state['draft']}\n\n"
                f"审核建议：\n{state['review_suggestions']}\n\n"
                f"用户补充信息：{state.get('user_context') or '无'}\n"
                f"{supplemental_context}"
            )

        return await self._run_stage(
            state,
            stage="润色",
            primary_agent="polish",
            backup_agent="polish_backup",
            output_key="final_article",
            log_title="润色完成",
            build_input=build_input,
            min_chars=min_chars,
            max_chars=max_chars,
            include_body_in_log=False,
        )
