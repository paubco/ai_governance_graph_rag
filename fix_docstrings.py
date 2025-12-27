#!/usr/bin/env python3
"""Fix mismatched docstring quotes in all Python files."""

import re
from pathlib import Path

# List of all files that need fixing
files_to_fix = """config/extraction_config.py
scripts/run_graph_construction.py
scripts/run_query.py
config/retrieval_config.py
tests/utils/test_api_debug.py
tests/utils/test_embedder.py
tests/extraction/test_enrichment.py
tests/utils/__init__.py
tests/retrieval/test_metrics.py
tests/retrieval/test_parameter_tuning.py
tests/retrieval/test_retrieval_ablation.py
tests/ingestion/__init__.py
tests/ingestion/test_document_loader.py
tests/retrieval/test_answer_generator.py
tests/retrieval/test_retrieval_complete.py
tests/graph/__init__.py
tests/graph/faiss_builder_test.py
tests/graph/neo4j_import_test.py
tests/processing/test_relation_extraction_parallel.py
tests/processing/test_entity_disambiguator.py
tests/processing/test_entity_extraction.py
tests/processing/test_relation_extraction.py
tests/processing/test_semantic_chunker.py
src/preprocessing/tests/test_preprocessing.py
src/retrieval/tests/test_answer_generator.py
src/retrieval/tests/test_parameter_tuning.py
src/retrieval/tests/test_retrieval_complete.py
tests/processing/__init__.py
src/enrichment/tests/test_enrichment.py
src/graph/tests/test_graph.py
src/processing/chunks/tests/test_chunking.py
src/processing/relations/tests/test_relation_preflight.py
src/processing/entities/tests/test_disambiguation_quality.py
src/processing/entities/tests/test_disambiguation.py
src/processing/entities/tests/test_entity_extraction.py
src/processing/entities/tests/test_extraction_quality.py
src/processing/entities/tests/test_samejudge_preflight.py
src/processing/entities/tests/test_threshold_refinement.py
src/processing/relations/relation_extractor.py
src/processing/relations/__init__.py
src/processing/relations/build_entity_cooccurrence.py
src/processing/relations/relation_processor.py
src/processing/relations/validate_relations.py
src/processing/entities/pre_entity_extractor.py
src/processing/entities/pre_entity_filter.py
src/processing/entities/pre_entity_processor.py
src/processing/entities/semantic_disambiguator.py
src/processing/entities/__init__.py
src/processing/entities/disambiguation_processor.py
src/processing/entities/metadata_disambiguator.py
src/retrieval/chunk_retriever.py
src/retrieval/answer_generator.py
src/retrieval/entity_resolver.py
src/retrieval/graph_expander.py
src/retrieval/query_parser.py
src/retrieval/result_ranker.py
src/retrieval/retrieval_processor.py
src/analysis/ablation_study.py
src/analysis/retrieval_metrics.py
src/retrieval/__init__.py
src/analysis/ablation_latex_export.py
src/analysis/test_queries.py
src/analysis/__init__.py
src/analysis/graph_analytics.py
src/preprocessing/preprocessing_processor.py
src/preprocessing/text_cleaner.py
src/preprocessing/translator.py
src/prompts/prompts.py
src/enrichment/jurisdiction_matcher.py
src/enrichment/citation_matcher.py
src/enrichment/enrichment_processor.py
src/enrichment/__init__.py
src/enrichment/scopus_parser.py
src/ingestion/dlapiper_scraper.py
src/ingestion/document_loader.py
src/ingestion/paper_to_scopus_metadata_matcher.py
src/graph/faiss_builder.py
src/graph/neo4j_importer.py
src/ingestion/__init__.py
src/graph/neo4j_import_processor.py
src/__init__.py
src/graph/__init__.py
src/processing/__init__.py""".strip().split('\n')

project_root = Path('/home/paubco/projects/Graph_Rag')
fixed_count = 0

for file_path in files_to_fix:
    full_path = project_root / file_path
    if not full_path.exists():
        print(f"⚠️  Skipping {file_path} (not found)")
        continue

    content = full_path.read_text()

    # Fix the pattern: """\n""" -> just remove the duplicate
    # Pattern: closing docstring followed immediately by opening docstring
    new_content = re.sub(r'"""\n"""', '"""', content)

    if content != new_content:
        full_path.write_text(new_content)
        fixed_count += 1
        print(f"✓ Fixed {file_path}")
    else:
        print(f"  No change needed: {file_path}")

print(f"\n✅ Fixed {fixed_count} files")
