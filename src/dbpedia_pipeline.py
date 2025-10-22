import requests
from utils.dbpedia_cleaner import is_valid_entity
from utils.logger import logger, cleaning_logger

# ----------------------------------------
# DBpedia Spotlight Annotation
# ----------------------------------------
def annotate_text_spotlight(scopus_id, chunk_id, text, confidence=0.50, support=0):
    """
    Annotate a text using DBpedia Spotlight.
    Returns list of tuples: (scopus_id, chunk_id, URI, surface_form, similarity_score)
    """
    url = "https://api.dbpedia-spotlight.org/en/annotate"
    params = {"text": text, "confidence": confidence, "support": support}
    headers = {"Accept": "application/json"}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        annotations = response.json().get("Resources", [])
        logger.debug(f"Annotated {len(annotations)} entities from text.")
        return [
            (scopus_id, chunk_id, a["@URI"], a["@surfaceForm"], float(a["@similarityScore"]))
            for a in annotations
        ]
    except Exception as e:
        logger.error(f"Spotlight annotation failed: {e}")
        return []

# ----------------------------------------
# Fetch Entity JSON from DBpedia
# ----------------------------------------
def fetch_dbpedia_entity(uri):
    try:
        entity_name = uri.split("/")[-1]
        url = f"https://dbpedia.org/data/{entity_name}.json"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        key = f"http://dbpedia.org/resource/{entity_name}"
        return data.get(key, None)
    except Exception as e:
        logger.error(f"Failed to fetch entity {uri}: {e}")
        return None

# ----------------------------------------
# Parse DBpedia Entity
# ----------------------------------------
def parse_dbpedia_entity(entity_json):
    """
    Parses a DBpedia JSON entity into a structured dict, including relations and types.
    """
    result = {"label": None, "abstract": None, "types": [], "relations": {}, "wikidata_quids": []}

    def get_uri_values(prop):
        return [v["value"] for v in entity_json.get(prop, []) if v.get("type") == "uri"]

    # Label
    for item in entity_json.get("http://www.w3.org/2000/01/rdf-schema#label", []):
        if item.get("lang") == "en" or "lang" not in item:
            result["label"] = item["value"]
            break

    # Abstract
    abstracts = entity_json.get("http://dbpedia.org/ontology/abstract", []) or \
                entity_json.get("http://www.w3.org/2000/01/rdf-schema#comment", [])
    for item in abstracts:
        if item.get("lang") == "en":
            result["abstract"] = item["value"]
            break

    # Types
    result["types"] = [t["value"] for t in entity_json.get("http://www.w3.org/1999/02/22-rdf-syntax-ns#type", [])]

    # Wikidata QIDs
    sameas = get_uri_values("http://www.w3.org/2002/07/owl#sameAs")
    result["wikidata_quids"] = [s.rsplit("/", 1)[-1] for s in sameas if "wikidata.org/entity/" in s]

    # Categories
    categories = get_uri_values("http://purl.org/dc/terms/subject")
    if categories:
        result["relations"]["categories"] = categories

    return result

# ----------------------------------------
# Wrap raw Spotlight annotations into enriched parsed entities
# ----------------------------------------
def enrich_annotations(raw_annotations):
    """
    Takes raw Spotlight annotations [(article_id, chunk_id, URI, surface_form, score), ...] and returns
    parsed entity dicts with relations, types, Wikidata QIDs, surface form, and similarity score.

    Optimized: fetches and parses each unique URI only once, then merges back to all occurrences.
    """
    enriched = []
    if not raw_annotations:
        return enriched

    # 1 Extract unique URIs
    unique_uris = {uri for _, _, uri, _, _ in raw_annotations}

    # 2 Enrich only unique URIs
    uri_to_entity = {}
    for uri in unique_uris:
        entity_json = fetch_dbpedia_entity(uri)
        if entity_json:
            parsed = parse_dbpedia_entity(entity_json)
            parsed["uri"] = uri  # keep URI
            uri_to_entity[uri] = parsed
        else:
            logger.warning(f"No data for URI: {uri}")

    # 3 Merge enriched data back to all annotations
    for article_id, chunk_id, uri, surface_form, score in raw_annotations:
        entity_info = uri_to_entity.get(uri)
        if entity_info:
            entity_copy = entity_info.copy()  # avoid overwriting shared dict
            entity_copy.update({
                "scopus_id": article_id,
                "chunk_id": chunk_id,
                "surface_form": surface_form,
                "similarity_score": score
            })
            enriched.append(entity_copy)

    return enriched


# ----------------------------------------
# Clean & Validate Parsed Entities
# ----------------------------------------
def extract_and_clean_entities(entity_json_list):
    """
    Cleans and validates a list of parsed entity dicts.
    """
    entities_data = []
    rejected_count = 0

    for entity_info in entity_json_list:
        uri = entity_info.get("uri")
        label = entity_info.get("label")
        surface = entity_info.get("surface_form", "N/A")
        score = entity_info.get("similarity_score", 0)
        types = entity_info.get("types", [])

        is_valid, reason = is_valid_entity(label, types, score)
        if is_valid:
            cleaning_logger.info(f"✅ Accepted: {label} ({uri}) | Reason: {reason}")
            entities_data.append(entity_info)
        else:
            cleaning_logger.info(f"❌ Rejected: {label} ({uri}) | Reason: {reason}")
            rejected_count += 1

    cleaning_logger.info(f"Finished cleaning: {len(entities_data)} accepted, {rejected_count} rejected.")
    return entities_data