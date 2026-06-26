from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import query_engine


router = APIRouter()


class QueryRequest(BaseModel):
    question: str = Field(..., json_schema_extra={"example": "星辰科技的最大股东是谁？"})


class QueryResponse(BaseModel):
    final_answer: str
    reasoning_path: List[str]


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Receive a multi-hop QA query."""
    if not request.question:
        raise HTTPException(status_code=400, detail="Question must not be empty")

    try:
        return query_engine.multi_hop_query(request.question)
    except Exception as e:
        # Log details to the server for debugging.
        print(f"Error while processing the query: {e}")
        raise HTTPException(
            status_code=500, detail=f"Internal error while processing the query: {str(e)}"
        )
