import uvicorn
from fastapi import FastAPI

from . import config  # Ensure configuration is loaded and validated on startup.
from .api import router as api_router


app = FastAPI(
    title="GraphRAG Multi-hop QA System",
    description="An advanced QA API fusing document retrieval (RAG) and a knowledge graph (KG).",
    version="1.0.0",
)


app.include_router(api_router, prefix="/api")


@app.get("/")
def read_root():
    return {
        "message": (
            "Welcome to the GraphRAG API. First run 'python -m graph_rag.graph_builder' "
            "to build the knowledge graph, then visit /docs for the API documentation."
        )
    }


def main():
    # The assignment entry point goes here. Run from the project root with:
    #   python -m graph_rag.main
    print("Starting the FastAPI service...")
    print("Before starting, make sure you have built the graph:")
    print("  python -m graph_rag.graph_builder")
    print(f"Neo4j URI: {config.NEO4J_URI}")

    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()
