import os
import re

from llama_index.core import (
    StorageContext,
    SimpleDirectoryReader,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.prompts import PromptTemplate
from neo4j import GraphDatabase

from . import config


# Lazily-initialized singletons so importing this module never crashes when Neo4j
# is unreachable; engines are created on the first query instead.
_rag_query_engine = None
_graph_driver = None


# A compact description of the graph schema, handed to the LLM when it has to
# generate Cypher for open-ended questions (the "Cypher + LLM" collaboration).
GRAPH_SCHEMA = """
Node label: Entity
  - properties: name (string), type (string; one of '公司' / '机构' / '个人')
Relationship: (shareholder:Entity)-[:HOLDS_SHARES_IN {share_percentage: float}]->(company:Entity)
  - share_percentage is a number between 0 and 100.
""".strip()


def _get_rag_query_engine():
    """Build or load the RAG vector index over the company documents."""
    global _rag_query_engine
    if _rag_query_engine is not None:
        return _rag_query_engine

    if not os.path.exists(config.INDEX_DIR):
        print("No vector index found; building one from the document file...")
        documents = SimpleDirectoryReader(
            input_files=[config.COMPANY_DOC_PATH]
        ).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=config.INDEX_DIR)
        print(f"Vector index created and saved to '{config.INDEX_DIR}'.")
    else:
        print(f"Loading existing vector index from '{config.INDEX_DIR}'...")
        storage_context = StorageContext.from_defaults(persist_dir=config.INDEX_DIR)
        index = load_index_from_storage(storage_context)
        print("Vector index loaded.")

    _rag_query_engine = index.as_query_engine(similarity_top_k=2)
    return _rag_query_engine


def _get_graph_driver():
    """Connect to Neo4j (once) and return the native driver.

    We use the official neo4j driver instead of LlamaIndex's Neo4jGraphStore because
    the latter requires the APOC plugin (it calls apoc.meta.data on init). Our queries
    are plain Cypher, so the native driver keeps the system portable (no APOC needed).
    """
    global _graph_driver
    if _graph_driver is None:
        print("Connecting to Neo4j...")
        _graph_driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD),
        )
        _graph_driver.verify_connectivity()
    return _graph_driver


def _run_cypher(cypher: str) -> list:
    """Execute a Cypher query and return the rows as a list of dicts."""
    driver = _get_graph_driver()
    with driver.session(database=config.NEO4J_DATABASE) as session:
        return [record.data() for record in session.run(cypher)]


def _extract_entity(question: str) -> str:
    """Use the LLM to extract the core company/organization name from the question."""
    prompt = PromptTemplate(
        "Extract the single company or organization name from the question below.\n"
        "Return only the name, with no extra text or punctuation.\n"
        "Question: '{question}'"
    )
    response = config.Settings.llm.complete(prompt.format(question=question))
    return response.text.strip()


def _generate_cypher(question: str) -> str:
    """Ask the LLM to translate an open-ended question into a Cypher query."""
    prompt = PromptTemplate(
        "You are an expert at writing Neo4j Cypher queries.\n"
        "Given the graph schema and a question, return ONE valid Cypher query and nothing else.\n"
        "Do not wrap it in markdown fences.\n\n"
        "--- Graph schema ---\n{schema}\n\n"
        "--- Question ---\n{question}\n\n"
        "--- Cypher ---\n"
    )
    response = config.Settings.llm.complete(
        prompt.format(schema=GRAPH_SCHEMA, question=question)
    )
    cypher = response.text.strip()
    # Strip any accidental ```cypher ... ``` fences the model might add.
    cypher = re.sub(r"^```(?:cypher)?|```$", "", cypher, flags=re.MULTILINE).strip()
    return cypher


def multi_hop_query(question: str) -> dict:
    """
    Run a multi-hop query: entity recognition -> graph (KG) -> document (RAG) -> LLM.
    Returns the final answer plus a step-by-step reasoning path for explainability.
    """
    rag_query_engine = _get_rag_query_engine()

    reasoning_path = []

    # 1. Identify the core entity in the question.
    entity_name = _extract_entity(question)
    reasoning_path.append(
        f"Step 1: Identified the core entity in '{question}' -> '{entity_name}'"
    )

    # 2. Query the graph. For "largest shareholder" / "controlling" questions we use a
    #    deterministic Cypher template; otherwise we let the LLM generate the Cypher.
    if "最大股东" in question or "控股" in question:
        cypher_query = (
            "MATCH (shareholder:Entity)-[r:HOLDS_SHARES_IN]->"
            f"(company:Entity {{name: '{entity_name}'}}) "
            "RETURN shareholder.name AS shareholder, r.share_percentage AS percentage "
            "ORDER BY percentage DESC LIMIT 1"
        )
        reasoning_path.append(
            "Step 2: Detected the keyword '最大股东/控股'; built a precise Cypher template."
        )
    else:
        cypher_query = _generate_cypher(question)
        reasoning_path.append(
            "Step 2: No fixed pattern detected; the LLM generated a Cypher query (Cypher + LLM)."
        )

    reasoning_path.append(f"   - Cypher query: {cypher_query}")

    try:
        graph_response = _run_cypher(cypher_query)
        kg_result_text = str(graph_response)
    except Exception as e:
        # Guard against error propagation from a malformed LLM-generated query.
        kg_result_text = f"(graph query failed: {e})"
    reasoning_path.append(f"   - Graph result: {kg_result_text}")

    # 3. Retrieve background context for the entity via RAG.
    rag_response = rag_query_engine.query(
        f"Provide detailed background information about '{entity_name}'."
    )
    rag_context = "\n\n".join(
        node.get_content() for node in rag_response.source_nodes
    )
    reasoning_path.append(
        f"Step 3: Retrieved background context about '{entity_name}' via RAG."
    )
    reasoning_path.append(f"   - RAG context (truncated): {rag_context[:200]}...")

    # 4. Synthesize the final answer from both sources.
    final_answer_prompt = PromptTemplate(
        "You are a professional financial analyst. Answer the user's question clearly and "
        "concisely in the same language as the question, using only the information below. "
        "If the graph result is empty or contradicts the documents, say so instead of guessing.\n"
        "--- User question ---\n{question}\n\n"
        "--- Knowledge graph result ---\n{kg_result}\n\n"
        "--- Related documents ---\n{rag_context}\n\n"
        "--- Final answer ---\n"
    )
    reasoning_path.append(
        "Step 4: The LLM synthesizes the graph result and documents into a final answer."
    )
    final_response = config.Settings.llm.complete(
        final_answer_prompt.format(
            question=question, kg_result=kg_result_text, rag_context=rag_context
        )
    )

    return {
        "final_answer": final_response.text,
        "reasoning_path": reasoning_path,
    }
