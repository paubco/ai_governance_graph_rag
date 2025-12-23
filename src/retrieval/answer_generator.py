# -*- coding: utf-8 -*-
"""
Answer generator for AI governance GraphRAG pipeline.

Generates answers from retrieval results using Claude API with structured prompts,
token budgeting, and citation extraction.
"""

# Standard library
import os
from dataclasses import dataclass
from typing import Optional

# Third-party
from anthropic import Anthropic

# Config imports (direct)
from config.retrieval_config import ANSWER_GENERATION_CONFIG

# Dataclass imports (direct)
from src.utils.dataclasses import RetrievalResult

# Utils
from src.utils.token_counter import TokenCounter, count_tokens
from src.utils.logger import get_logger

# Prompts
from src.prompts.prompts import (
    ANSWER_GENERATION_SYSTEM_PROMPT,
    ANSWER_GENERATION_USER_PROMPT,
)

logger = get_logger(__name__)


@dataclass
class GeneratedAnswer:
    """LLM-generated answer with metadata."""
    answer: str
    query: str
    input_tokens: int
    output_tokens: int
    model: str
    cost_usd: float
    retrieval_chunks_used: int
    graph_entities_used: int
    graph_relations_used: int


class AnswerGenerator:
    """
    Generate answers from retrieval results using Claude API.
    
    Handles:
    - Token budgeting and prompt assembly
    - API calls to Claude
    - Cost tracking
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: dict = None
    ):
        """
        Initialize answer generator.
        
        Args:
            api_key: Anthropic API key (reads from env if None).
            config: Generation config (uses ANSWER_GENERATION_CONFIG if None).
        """
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        
        self.client = Anthropic(api_key=self.api_key)
        self.config = config or ANSWER_GENERATION_CONFIG
        self.token_counter = TokenCounter()
        
        logger.info("AnswerGenerator initialized with model: %s", self.config['model'])
    
    def generate(self, retrieval_result: RetrievalResult) -> GeneratedAnswer:
        """
        Generate answer from retrieval result.
        
        Args:
            retrieval_result: Output from retrieval pipeline.
            
        Returns:
            GeneratedAnswer with answer text and metadata.
        """
        logger.info("Generating answer for query: %s", retrieval_result.query)
        
        formatted = self._format_prompt_with_budget(retrieval_result)
        
        messages = [{"role": "user", "content": formatted['user_prompt']}]
        
        response = self.client.messages.create(
            model=self.config['model'],
            max_tokens=self.config['max_output_tokens'],
            temperature=self.config['temperature'],
            top_p=self.config['top_p'],
            system=ANSWER_GENERATION_SYSTEM_PROMPT,
            messages=messages
        )
        
        answer_text = response.content[0].text
        output_tokens = response.usage.output_tokens
        input_tokens = response.usage.input_tokens
        
        cost = self.token_counter.estimate_cost(
            input_tokens, output_tokens, model="claude-haiku"
        )
        
        logger.info(
            "Answer generated: %d input tokens, %d output tokens, $%.4f cost",
            input_tokens, output_tokens, cost
        )
        
        return GeneratedAnswer(
            answer=answer_text,
            query=retrieval_result.query,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.config['model'],
            cost_usd=cost,
            retrieval_chunks_used=len(formatted['chunks_included']),
            graph_entities_used=len(retrieval_result.subgraph.entity_ids) if retrieval_result.subgraph else 0,
            graph_relations_used=len(retrieval_result.subgraph.relations) if retrieval_result.subgraph else 0,
        )
    
    def _format_prompt_with_budget(self, retrieval_result: RetrievalResult) -> dict:
        """Format retrieval result into prompt with token budgeting."""
        budget = self.config['token_budget']
        
        graph_str = self._format_graph_structure(
            retrieval_result.subgraph, max_tokens=budget['graph_structure']
        )
        
        entity_str = self._format_entity_context(
            retrieval_result.subgraph, max_tokens=budget['entity_context']
        )
        
        sources_str, chunks_included = self._format_sources_with_budget(
            retrieval_result.chunks, max_tokens=budget['source_chunks']
        )
        
        user_prompt = ANSWER_GENERATION_USER_PROMPT.format(
            query=retrieval_result.query,
            graph_structure=graph_str,
            entity_context=entity_str,
            sources=sources_str
        )
        
        return {
            'user_prompt': user_prompt,
            'chunks_included': chunks_included,
            'tokens_used': {
                'graph': count_tokens(graph_str),
                'entities': count_tokens(entity_str),
                'sources': count_tokens(sources_str),
                'total': count_tokens(user_prompt)
            }
        }
    
    def _format_graph_structure(self, subgraph, max_tokens: int) -> str:
        """Format graph relations for prompt."""
        if not subgraph or not subgraph.relations:
            return "No relations found."
        
        lines = []
        tokens_used = 0
        
        for rel in subgraph.relations:
            line = f"- {rel.subject_id} --{rel.predicate}--> {rel.object_id}"
            line_tokens = count_tokens(line)
            
            if tokens_used + line_tokens > max_tokens:
                break
            
            lines.append(line)
            tokens_used += line_tokens
        
        return "\n".join(lines) if lines else "No relations (budget exceeded)."
    
    def _format_entity_context(self, subgraph, max_tokens: int) -> str:
        """Format key entities for prompt."""
        if not subgraph or not subgraph.entity_ids:
            return "No entities found."
        
        lines = []
        tokens_used = 0
        
        for eid in list(subgraph.entity_ids)[:20]:
            line = f"- {eid}"
            line_tokens = count_tokens(line)
            
            if tokens_used + line_tokens > max_tokens:
                break
            
            lines.append(line)
            tokens_used += line_tokens
        
        return "\n".join(lines) if lines else "Entities present but budget exceeded."
    
    def _format_sources_with_budget(self, chunks, max_tokens: int) -> tuple:
        """Format source chunks with token budget."""
        if not chunks:
            return "No sources retrieved.", []
        
        lines = []
        chunks_included = []
        tokens_used = 0
        max_chunks = self.config['max_chunks_to_format']
        
        for i, chunk in enumerate(chunks[:max_chunks], 1):
            chunk_text = chunk.text
            max_chunk_chars = self.config.get('truncate_chunk_chars', 1000)
            if len(chunk_text) > max_chunk_chars:
                chunk_text = chunk_text[:max_chunk_chars] + "..."
            
            source_line = f"[{i}] Source: {chunk.doc_id} (Type: {chunk.doc_type}, Jurisdiction: {chunk.jurisdiction or 'N/A'})"
            full_chunk = f"{source_line}\n{chunk_text}\n"
            
            chunk_tokens = count_tokens(full_chunk)
            
            if tokens_used + chunk_tokens > max_tokens:
                break
            
            lines.append(full_chunk)
            chunks_included.append(chunk)
            tokens_used += chunk_tokens
        
        return "\n".join(lines) if lines else "Sources retrieved but budget exceeded.", chunks_included
    
    def estimate_cost_for_query(self, retrieval_result: RetrievalResult) -> dict:
        """Estimate cost before actually generating."""
        formatted = self._format_prompt_with_budget(retrieval_result)
        
        input_tokens = (
            count_tokens(ANSWER_GENERATION_SYSTEM_PROMPT) +
            count_tokens(formatted['user_prompt'])
        )
        output_tokens = self.config['max_output_tokens']
        
        cost = self.token_counter.estimate_cost(
            input_tokens, output_tokens, model="claude-haiku"
        )
        
        return {
            'estimated_input_tokens': input_tokens,
            'estimated_output_tokens': output_tokens,
            'estimated_cost_usd': cost,
            'chunks_that_fit': len(formatted['chunks_included']),
            'total_chunks_available': len(retrieval_result.chunks)
        }