import pandas as pd
from neo4j import GraphDatabase

from . import config


def build_graph():
    """Read data from the CSV file and build the Neo4j knowledge graph."""
    driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD),
    )

    df = pd.read_csv(config.SHAREHOLDER_CSV_PATH)

    with driver.session(database=config.NEO4J_DATABASE) as session:
        # Clear the database to avoid duplicate creation.
        print("Clearing existing graph data...")
        session.run("MATCH (n) DETACH DELETE n")

        print("Creating nodes and relationships...")
        # Batch-create with UNWIND for efficiency.
        # 1. Create all company and shareholder nodes.
        query_create_nodes = """
        UNWIND $rows AS row
        MERGE (c:Entity {name: row.company_name})
        ON CREATE SET c.type = '公司'
        MERGE (s:Entity {name: row.shareholder_name})
        ON CREATE SET s.type = row.shareholder_type
        """
        session.run(query_create_nodes, rows=df.to_dict("records"))

        # 2. Create the shareholding relationships.
        query_create_rels = """
        UNWIND $rows AS row
        MATCH (shareholder:Entity {name: row.shareholder_name})
        MATCH (company:Entity {name: row.company_name})
        MERGE (shareholder)-[r:HOLDS_SHARES_IN]->(company)
        SET r.share_percentage = toFloat(row.share_percentage)
        """
        session.run(query_create_rels, rows=df.to_dict("records"))

        print("Graph nodes and relationships created.")

        # Create an index to optimize query performance.
        print("Creating an index on the 'name' property of 'Entity' nodes...")
        try:
            session.run(
                "CREATE INDEX entity_name_index IF NOT EXISTS FOR (n:Entity) ON (n.name)"
            )
            print("Index created successfully.")
        except Exception as e:
            print(f"Error creating index: {e}")

    driver.close()
    print("Graph build process finished.")


if __name__ == "__main__":
    build_graph()
