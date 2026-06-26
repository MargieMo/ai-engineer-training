import os

import pandas as pd
from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.vector_stores.milvus import MilvusVectorStore

from . import config


# Singleton-style module state so there is only one index / query engine globally.
_query_engine = None
_index = None


def _make_vector_store(overwrite: bool) -> MilvusVectorStore:
    return MilvusVectorStore(
        uri=config.MILVUS_URI,
        collection_name=config.COLLECTION_NAME,
        dim=config.DIMENSION,
        overwrite=overwrite,
        # With the default output_fields=["*"], Milvus Lite returns only the primary
        # key `id` and drops the `text` column, so reconstructed nodes have empty text
        # (and the API responds with "答案未找到"). Passing an explicit output_fields
        # makes LlamaIndex request the `text` field so node content is restored.
        output_fields=["doc_id"],
    )


def _initialize_index():
    """Initialize the index, either by building from CSV or loading an existing collection."""
    global _index, _query_engine

    print("Initializing Milvus vector store...")

    # No local Milvus Lite data file yet means this is the first run -> build from CSV.
    first_run = not os.path.exists(config.MILVUS_URI)

    vector_store = _make_vector_store(overwrite=False)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if first_run and os.path.exists(config.FAQ_FILE):
        print("First run: building index from the CSV file.")
        _index = _build_index_from_file(storage_context)
    elif not os.path.exists(config.FAQ_FILE):
        print(f"Warning: data file {config.FAQ_FILE} not found. Creating an empty index.")
        _index = VectorStoreIndex.from_documents([], storage_context=storage_context)
    else:
        # Reuse the already-persisted Milvus collection.
        # Milvus Lite does not auto-load an existing collection into memory, so the
        # first search raises "Collection ... is in state 'released'". We therefore
        # load it explicitly, falling back to a rebuild from CSV if loading fails
        # (e.g. corrupted collection or dimension mismatch).
        print("Loading index from the existing Milvus collection.")
        try:
            vector_store.client.load_collection(config.COLLECTION_NAME)
            _index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        except Exception as e:
            print(f"Failed to load existing collection ({e}); rebuilding from CSV.")
            rebuild_store = _make_vector_store(overwrite=True)
            rebuild_ctx = StorageContext.from_defaults(vector_store=rebuild_store)
            _index = _build_index_from_file(rebuild_ctx)

    _query_engine = _index.as_query_engine(similarity_top_k=3)
    print("Index and query engine are ready.")


def _build_index_from_file(storage_context: StorageContext) -> VectorStoreIndex:
    """Build the index from the CSV file."""
    print(f"Building a new index from {config.FAQ_FILE}...")
    df = pd.read_csv(config.FAQ_FILE)
    documents = []
    for _, row in df.iterrows():
        doc_text = f"问题: {row['question']}\n答案: {row['answer']}"
        documents.append(
            Document(text=doc_text, metadata={"question": row["question"]})
        )

    # Semantic splitter to optimize chunking (semantic split + overlap).
    splitter = SemanticSplitterNodeParser.from_defaults(embed_model=config.EMBED_MODEL)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
    )
    return index


def get_query_engine():
    """Public accessor for the query engine."""
    if _query_engine is None:
        _initialize_index()
    return _query_engine


def update_index():
    """
    Hot-reload the index.
    Clears the existing collection and rebuilds it from the file (auto re-index).
    """
    global _index, _query_engine
    print("Starting index hot-reload...")

    # Create a fresh Milvus store with overwrite=True to clear the old collection.
    vector_store = _make_vector_store(overwrite=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    _index = _build_index_from_file(storage_context)
    _query_engine = _index.as_query_engine(similarity_top_k=3)

    print("Index hot-reload complete.")
    return {"message": "Index updated successfully."}


# Initialize automatically when the module is imported.
_initialize_index()
