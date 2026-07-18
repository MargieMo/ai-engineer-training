"""Episodic memory + FAISS semantic memory (RAG)."""

from __future__ import annotations

from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from . import config
from .models import SpeechRecord


class GameMemory:
    """Dual memory: structured episodic events + semantic speech retrieval."""

    def __init__(self) -> None:
        self.episodic: list[str] = []
        self._embeddings = OpenAIEmbeddings(
            model=config.OPENAI_EMBEDDING_MODEL,
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_API_BASE,
        )
        self._store: FAISS | None = None
        self._speech_count = 0

    def add_event(self, event: str) -> None:
        self.episodic.append(event)

    def situation_summary(self, limit: int = 12) -> str:
        if not self.episodic:
            return "（尚无历史事件）"
        return "\n".join(f"- {e}" for e in self.episodic[-limit:])

    def add_speech(self, speech: SpeechRecord) -> None:
        text = (
            f"第{speech['round']}天 {speech['speaker']} 说：{speech['content']}"
        )
        self.add_event(text)
        doc = Document(
            page_content=text,
            metadata={
                "round": speech["round"],
                "speaker": speech["speaker"],
            },
        )
        if self._store is None:
            self._store = FAISS.from_documents([doc], self._embeddings)
        else:
            self._store.add_documents([doc])
        self._speech_count += 1

    def retrieve(self, query: str, k: int | None = None) -> str:
        """RAG: retrieve semantically similar historical speeches."""
        if self._store is None or self._speech_count == 0:
            return "（暂无历史发言可检索）"
        top_k = min(k or config.RAG_TOP_K, self._speech_count)
        docs = self._store.similarity_search(query, k=top_k)
        if not docs:
            return "（未检索到相关发言）"
        return "\n".join(f"- {d.page_content}" for d in docs)

    def today_speeches(self, speeches: list[SpeechRecord], round_no: int) -> str:
        today = [s for s in speeches if s["round"] == round_no]
        if not today:
            return "（今日尚无发言）"
        return "\n".join(f"- {s['speaker']}: {s['content']}" for s in today)
