import uvicorn
from fastapi import FastAPI

from . import config  # Ensure configuration is loaded and validated on startup.
from .api import router as api_router


app = FastAPI(
    title="Milvus FAQ Retrieval System",
    description="A FAQ Q&A API built on LlamaIndex and Milvus (OpenAI embeddings).",
    version="1.0.0",
)


app.include_router(api_router, prefix="/api")


@app.get("/")
def read_root():
    return {"message": "Welcome to the Milvus FAQ Retrieval System API. Visit /docs for details."}


def main():
    # The assignment entry point goes here. Run from the project root with:
    #   python -m milvus_faq.main
    print("Starting the FastAPI service...")
    print(f"Data file path: {config.FAQ_FILE}")
    print(f"Milvus Lite database path: {config.MILVUS_URI}")
    print(f"Embedding model: {config.EMBED_MODEL.model_name}")

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
