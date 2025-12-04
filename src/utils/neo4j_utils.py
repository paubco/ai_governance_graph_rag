# -*- coding: utf-8 -*-
from neo4j import GraphDatabase
import ast

def get_driver(uri, user, password):
    return GraphDatabase.driver(uri, auth=(user, password))

def create_publication_graph(tx, row):
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
    with driver.session() as session:
        for _, row in metadata_df.iterrows():
            session.execute_write(create_publication_graph, row)

def close_driver(driver):
    if driver:
        driver.close()