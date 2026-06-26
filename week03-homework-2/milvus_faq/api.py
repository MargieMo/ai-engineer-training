from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import index_manager


router = APIRouter()


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    score: float


@router.post("/query", response_model=list[QueryResponse])
async def query_faq(request: QueryRequest):
    """Receive a user question and return the most relevant FAQ entries."""
    if not request.question:
        raise HTTPException(status_code=400, detail="Question must not be empty")

    print(f"Received query: {request.question}")
    query_engine = index_manager.get_query_engine()
    response = query_engine.query(request.question)

    if not response.source_nodes:
        return []

    results = []
    for node in response.source_nodes:
        # Parse the original question and answer back out of the node text.
        text_parts = node.get_text().split("\n答案: ")
        original_question = text_parts[0].replace("问题: ", "")
        answer = text_parts[1] if len(text_parts) > 1 else "答案未找到"

        results.append(
            QueryResponse(
                question=original_question,
                answer=answer,
                score=node.get_score() or 0.0,
            )
        )

    return results


@router.post("/update-index")
async def update_faq_index():
    """
    Trigger a hot-reload of the knowledge base.
    The system reloads from data/faqs.csv and rebuilds the index.
    """
    try:
        result = index_manager.update_index()
        return {"status": "success", "message": result["message"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Index update failed: {str(e)}")
