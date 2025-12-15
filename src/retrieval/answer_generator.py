# -*- coding: utf-8 -*-
"""
Answer generator for AI governance GraphRAG pipeline.

Generates answers from retrieval results using Claude API with structured prompts,
token budgeting, and citation extraction. Formats retrieved chunks into context
and calls Claude with system/user prompts for answer synthesis.

Components:
    - Context formatting: Structures chunks with provenance metadata
    - Token budgeting: Ensures context fits within model limits
    - API interaction: Calls Claude API with retry handling
    - Citation extraction: Parses chunk references from generated answers

Example:
    generator = AnswerGenerator(api_key="...")
    answer = generator.generate(retrieval_result)
    print(answer.answer)  # LLM-generated response with citations
"""

# Standard library
import os
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
from anthropic import Anthropic

# Local
from src.retrieval.config import (
    RetrievalResult,
    ANSWER_GENERATION_CONFIG
)
from src.utils.token_counter import TokenCounter, count_tokens
from src.prompts.prompts import (
    ANSWER_GENERATION_SYSTEM_PROMPT,
    ANSWER_GENERATION_USER_PROMPT
)
from src.utils.logger import get_logger

# Setup logging
logger = get_logger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class GeneratedAnswer:
    """LLM-generated answer with metadata."""
    answer: str
    query: str
    input_tokens: int
    output_tokens: int
    model: str
    cost_usd: float
    
    # For analysis
    retrieval_chunks_used: int
    graph_entities_used: int
    graph_relations_used: int


# ============================================================================
# ANSWER GENERATOR
# ============================================================================

class AnswerGenerator:
    """
    Generate answers from retrieval results using Claude API.
    
    Handles:
    - Token budgeting and prompt assembly
    - API calls to Claude
    - Cost tracking
    - Citation extraction
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
        # API client
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        
        self.client = Anthropic(api_key=self.api_key)
        
        # Configuration
        self.config = config or ANSWER_GENERATION_CONFIG
        
        # Token counter
        self.token_counter = TokenCounter()
        
        logger.info("AnswerGenerator initialized with model: %s", self.config['model'])
    
    def generate(
        self,
        retrieval_result: RetrievalResult
    ) -> GeneratedAnswer:
        """
        Generate answer from retrieval result.
        
        Args:
            retrieval_result: Output from retrieval pipeline.
            
        Returns:
            GeneratedAnswer with answer text and metadata.
        """
        logger.info("Generating answer for query: %s", retrieval_result.query)
        
        # Format prompt with token budgeting
        formatted = self._format_prompt_with_budget(retrieval_result)
        logger.debug("Prompt formatted: %d total tokens", formatted['tokens_used']['total'])
        
        # Build messages
        messages = [
            {
                "role": "user",
                "content": formatted['user_prompt']
            }
        ]
        
        # Count input tokens
        input_tokens = (
            count_tokens(ANSWER_GENERATION_SYSTEM_PROMPT) +
            count_tokens(formatted['user_prompt'])
        )
        
        # Call Claude API
        logger.info("Calling Claude API (model: %s)", self.config['model'])
        response = self.client.messages.create(
            model=self.config['model'],
            max_tokens=self.config['max_output_tokens'],
            temperature=self.config['temperature'],
            top_p=self.config['top_p'],
            system=ANSWER_GENERATION_SYSTEM_PROMPT,
            messages=messages
        )
        
        # Extract answer
        answer_text = response.content[0].text
        
        # Get actual token usage from response
        output_tokens = response.usage.output_tokens
        input_tokens_actual = response.usage.input_tokens
        
        # Estimate cost
        cost = self.token_counter.estimate_cost(
            input_tokens_actual,
            output_tokens,
            model="claude-haiku"
        )
        
        logger.info(
            "Answer generated: %d input tokens, %d output tokens, $%.4f cost",
            input_tokens_actual, output_tokens, cost
        )
        
        return GeneratedAnswer(
            answer=answer_text,
            query=retrieval_result.query,
            input_tokens=input_tokens_actual,
            output_tokens=output_tokens,
            model=self.config['model'],
            cost_usd=cost,
            retrieval_chunks_used=len(formatted['chunks_included']),
            graph_entities_used=len(retrieval_result.subgraph.entities),
            graph_relations_used=len(retrieval_result.subgraph.relations)
        )
    
    def _format_prompt_with_budget(
        self,
        retrieval_result: RetrievalResult
    ) -> dict:
        """
        Format retrieval result into prompt with token budgeting.
        
        Args:
            retrieval_result: Complete retrieval output.
            
        Returns:
            Dict with user_prompt, chunks_included, and tokens_used.
        """
        budget = self.config['token_budget']
        
        # Format graph structure
        graph_str = self._format_graph_structure(
            retrieval_result.subgraph,
            max_tokens=budget['graph_structure']
        )
        
        # Format entity context
        entity_str = self._format_entity_context(
            retrieval_result.subgraph,
            max_tokens=budget['entity_context']
        )
        
        # Format sources with remaining budget
        sources_str, chunks_included = self._format_sources_with_budget(
            retrieval_result.chunks,
            max_tokens=budget['source_chunks']
        )
        
        # Assemble user prompt
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
        """
        Format graph relations for prompt.
        
        Args:
            subgraph: PCST subgraph with relations.
            max_tokens: Token budget for this section.
            
        Returns:
            Formatted string with relations as bullets.
        """
        if not subgraph.relations:
            return "No relations found."
        
        lines = []
        tokens_used = 0
        
        for rel in subgraph.relations:
            # Format: source --predicate--> target
            line = f"• {rel.source_name} --{rel.predicate}--> {rel.target_name}"
            line_tokens = count_tokens(line)
            
            if tokens_used + line_tokens > max_tokens:
                break
            
            lines.append(line)
            tokens_used += line_tokens
        
        if not lines:
            return "No relations (budget exceeded)."
        
        logger.debug("Formatted %d relations (%d tokens)", len(lines), tokens_used)
        return "\n".join(lines)
    
    def _format_entity_context(self, subgraph, max_tokens: int) -> str:
        """
        Format key entities for prompt.
        
        Args:
            subgraph: PCST subgraph with entities.
            max_tokens: Token budget for this section.
            
        Returns:
            Formatted string with entity list.
        """
        if not subgraph.entities:
            return "No entities found."
        
        # Take first N entities that fit budget
        lines = []
        tokens_used = 0
        
        for eid in list(subgraph.entities)[:20]:  # Limit to top 20
            line = f"• {eid}"
            line_tokens = count_tokens(line)
            
            if tokens_used + line_tokens > max_tokens:
                break
            
            lines.append(line)
            tokens_used += line_tokens
        
        if not lines:
            return "Entities present but budget exceeded."
        
        logger.debug("Formatted %d entities (%d tokens)", len(lines), tokens_used)
        return "\n".join(lines)
    
    def _format_sources_with_budget(
        self,
        chunks,
        max_tokens: int
    ) -> tuple:
        """
        Format source chunks with token budget.
        
        Args:
            chunks: List of RankedChunk objects.
            max_tokens: Token budget for sources section.
            
        Returns:
            Tuple of (formatted_sources_str, list_of_chunks_included).
        """
        if not chunks:
            return "No sources retrieved.", []
        
        lines = []
        chunks_included = []
        tokens_used = 0
        max_chunks = self.config['max_chunks_to_format']
        
        for i, chunk in enumerate(chunks[:max_chunks], 1):
            # Truncate individual chunk if needed
            chunk_text = chunk.text
            max_chunk_chars = self.config.get('truncate_chunk_chars', 1000)
            if len(chunk_text) > max_chunk_chars:
                chunk_text = chunk_text[:max_chunk_chars] + "..."
            
            # Format: [1] Source: ... \n Text
            source_line = f"[{i}] Source: {chunk.doc_id} (Type: {chunk.doc_type}, Jurisdiction: {chunk.jurisdiction or 'N/A'})"
            text_line = chunk_text
            full_chunk = f"{source_line}\n{text_line}\n"
            
            chunk_tokens = count_tokens(full_chunk)
            
            # Stop if budget exceeded
            if tokens_used + chunk_tokens > max_tokens:
                logger.debug("Token budget exceeded at chunk %d", i)
                break
            
            lines.append(full_chunk)
            chunks_included.append(chunk)
            tokens_used += chunk_tokens
        
        if not lines:
            return "Sources retrieved but budget exceeded (increase token budget).", []
        
        logger.debug("Formatted %d chunks (%d tokens)", len(chunks_included), tokens_used)
        return "\n".join(lines), chunks_included
    
    def estimate_cost_for_query(
        self,
        retrieval_result: RetrievalResult
    ) -> dict:
        """
        Estimate cost before actually generating.
        
        Useful for budget planning.
        
        Args:
            retrieval_result: Complete retrieval output.
            
        Returns:
            Dict with estimated input/output tokens and cost.
        """
        formatted = self._format_prompt_with_budget(retrieval_result)
        
        input_tokens = (
            count_tokens(ANSWER_GENERATION_SYSTEM_PROMPT) +
            count_tokens(formatted['user_prompt'])
        )
        
        output_tokens = self.config['max_output_tokens']
        
        cost = self.token_counter.estimate_cost(
            input_tokens,
            output_tokens,
            model="claude-haiku"
        )
        
        logger.info(
            "Cost estimate: %d input + %d output tokens = $%.4f",
            input_tokens, output_tokens, cost
        )
        
        return {
            'estimated_input_tokens': input_tokens,
            'estimated_output_tokens': output_tokens,
            'estimated_cost_usd': cost,
            'chunks_that_fit': len(formatted['chunks_included']),
            'total_chunks_available': len(retrieval_result.chunks)
        }