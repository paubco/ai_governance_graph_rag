# -*- coding: utf-8 -*-
"""
Neo4j database utilities for academic publication graph construction.

Helper functions for connecting to Neo4j and loading academic publication
metadata into graph database. Creates nodes and relationships for publications,
authors, journals, and affiliations.

Note:
    Currently unused in active pipeline - reserved for future Phase 2 integration.

Graph schema:
    - (Publication) - [PUBLISHED_IN] -> (Journal)
    - (Author) - [WROTE] -> (Publication)
    - (Author) - [AFFILIATED_WITH] -> (Affiliation)
"""
from neo4j import GraphDatabase
import ast


def get_driver(uri, user, password):
    """
    Create Neo4j driver instance.

    Args:
        uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
        user: Database username
        password: Database password

    Returns:
        neo4j.Driver: Authenticated Neo4j driver instance

    Example:
        >>> driver = get_driver("bolt://localhost:7687", "neo4j", "password")
    """
    return GraphDatabase.driver(uri, auth=(user, password))


def create_publication_graph(tx, row):
    """
    Create graph nodes and relationships for a single publication.

    Executes Cypher queries to create/update Publication, Author, Journal,
    and Affiliation nodes with appropriate relationships. Uses MERGE to
    avoid duplicates.

    Args:
        tx: Neo4j transaction object
        row: DataFrame row with publication metadata
            Required fields: scopus_id, title, creator, publication_name
            Optional fields: cover_date, subtype_desc, links, affiliations

    Note:
        This is a transaction function - call via session.execute_write()

    Example:
        >>> session.execute_write(create_publication_graph, df_row)
    """
    affiliations = row["affiliations"]
    if isinstance(affiliations, str):
        try:
            affiliations = ast.literal_eval(affiliations)
        except Exception:
            affiliations = []
    if isinstance(affiliations, dict):
        affiliations = [affiliations]

    tx.run(
        """
        MERGE (p:Publication {scopus_id: $scopus_id})
        SET p.title = $title,
            p.cover_date = $cover_date,
            p.subtype_desc = $subtype_desc,
            p.link_self = $link_self
        """,
        scopus_id=row["scopus_id"],
        title=row["title"],
        cover_date=row["cover_date"],
        subtype_desc=row["subtype_desc"],
        link_self=row["links"].get("self") if isinstance(row["links"], dict) else None
    )

    tx.run(
        """
        MERGE (a:Author {name: $creator})
        MERGE (p:Publication {scopus_id: $scopus_id})
        MERGE (a)-[:WROTE]->(p)
        """,
        creator=row["creator"],
        scopus_id=row["scopus_id"]
    )

    tx.run(
        """
        MERGE (j:Journal {name: $journal})
        MERGE (p:Publication {scopus_id: $scopus_id})
        MERGE (p)-[:PUBLISHED_IN]->(j)
        """,
        journal=row["publication_name"],
        scopus_id=row["scopus_id"]
    )

    for aff in affiliations:
        name = aff.get("name") if isinstance(aff, dict) else aff
        if name:
            tx.run(
                """
                MERGE (f:Affiliation {name: $aff_name})
                MERGE (a:Author {name: $creator})
                MERGE (a)-[:AFFILIATED_WITH]->(f)
                """,
                aff_name=name,
                creator=row["creator"]
            )


def load_metadata_to_neo4j(driver, metadata_df):
    """
    Load all publications from DataFrame into Neo4j.

    Iterates through DataFrame rows and creates graph nodes/relationships
    for each publication using write transactions.

    Args:
        driver: Neo4j driver instance (from get_driver())
        metadata_df: pandas DataFrame with publication metadata
            Must contain columns: scopus_id, title, creator, publication_name

    Example:
        >>> driver = get_driver("bolt://localhost:7687", "neo4j", "password")
        >>> load_metadata_to_neo4j(driver, publications_df)
        >>> close_driver(driver)
    """
    with driver.session() as session:
        for _, row in metadata_df.iterrows():
            session.execute_write(create_publication_graph, row)


def close_driver(driver):
    """
    Close Neo4j driver connection.

    Args:
        driver: Neo4j driver instance to close (can be None)

    Example:
        >>> close_driver(driver)
    """
    if driver:
        driver.close()
