# -*- coding: utf-8 -*-
"""
Evaluation

Provides:
- metrics.py: Dataclasses for entity resolution, retrieval, RAGAS, etc.
- ablation_study.py: Comparative analysis across retrieval modes

"""
from src.analysis.retrieval_metrics import (
    EntityResolutionMetrics,
    GraphUtilizationMetrics,
    CoverageMetrics,
    RetrievalMetrics,
    RAGASMetrics,
    PerformanceMetrics,
    TestResult,
    compute_entity_resolution_metrics,
    compute_graph_utilization_metrics,
    compute_retrieval_metrics,
    compute_coverage_metrics,
)

__all__ = [
    'EntityResolutionMetrics',
    'GraphUtilizationMetrics',
    'CoverageMetrics',
    'RetrievalMetrics',
    'RAGASMetrics',
    'PerformanceMetrics',
    'TestResult',
    'compute_entity_resolution_metrics',
    'compute_graph_utilization_metrics',
    'compute_retrieval_metrics',
    'compute_coverage_metrics',
]