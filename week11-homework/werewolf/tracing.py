"""Thought / Action / Observation tracing and LLM cost tracking."""

from __future__ import annotations

import time
from typing import Any

from langchain_openai import ChatOpenAI

from . import config
from .models import CostStats, TraceRecord


def empty_cost() -> CostStats:
    return {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "total_latency_ms": 0.0,
        "latencies_ms": [],
    }


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=config.OPENAI_MODEL,
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_API_BASE,
        temperature=0.7,
    )


def record_usage(cost: CostStats, response: Any, latency_ms: float) -> None:
    usage = getattr(response, "usage_metadata", None) or {}
    prompt = int(usage.get("input_tokens") or 0)
    completion = int(usage.get("output_tokens") or 0)
    total = int(usage.get("total_tokens") or (prompt + completion))
    cost["calls"] += 1
    cost["prompt_tokens"] += prompt
    cost["completion_tokens"] += completion
    cost["total_tokens"] += total
    cost["total_latency_ms"] += latency_ms
    cost["latencies_ms"].append(latency_ms)


def invoke_llm(llm: ChatOpenAI, messages: list[Any], cost: CostStats) -> str:
    started = time.perf_counter()
    response = llm.invoke(messages)
    latency_ms = (time.perf_counter() - started) * 1000
    record_usage(cost, response, latency_ms)
    return str(response.content)


def add_trace(
    traces: list[TraceRecord],
    *,
    round_no: int,
    phase: str,
    agent: str,
    thought: str,
    action: str,
    observation: str,
) -> None:
    traces.append(
        {
            "round": round_no,
            "phase": phase,
            "agent": agent,
            "thought": thought,
            "action": action,
            "observation": observation,
        }
    )


def summarize_cost(cost: CostStats) -> str:
    calls = cost["calls"] or 1
    avg_latency = cost["total_latency_ms"] / calls
    # Rough local-GPU estimate for a ~8B chat model serving similar QPS.
    # Assumes ~0.5–1.5 GPU-sec per call on a single mid-range GPU.
    gpu_seconds = calls * 1.0
    gpu_hours = gpu_seconds / 3600
    return (
        f"- LLM 调用次数: {cost['calls']}\n"
        f"- Prompt tokens: {cost['prompt_tokens']}\n"
        f"- Completion tokens: {cost['completion_tokens']}\n"
        f"- Total tokens: {cost['total_tokens']}\n"
        f"- 平均响应延迟: {avg_latency:.1f} ms\n"
        f"- 总延迟: {cost['total_latency_ms']:.1f} ms\n"
        f"- GPU 资源粗估（本地部署约 8B 模型）: "
        f"约 {gpu_seconds:.0f} GPU·秒 / {gpu_hours:.3f} GPU·小时"
    )
