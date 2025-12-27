# -*- coding: utf-8 -*-
"""
LaTeX exporter for ablation study results.

Generates:
1. ablation_table.tex - Results table grouped by category (both modes)
2. ablation_vars.tex - \\newcommand variables for inline citations (both modes)
3. ablation_appendix.tex - Detailed I/O per query (DETAILED MODE ONLY)
4. ablation_data.tex - Data for pgfplots figures (both modes)

Usage:
    from src.analysis.ablation_latex_export import LaTeXExporter
    exporter = LaTeXExporter(json_path, is_detailed=True)
    exporter.export_all(output_dir)
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    if not text:
        return ""
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
        '\\': r'\textbackslash{}',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def truncate_text(text: str, max_len: int = 80) -> str:
    """Truncate text for display."""
    if len(text) <= max_len:
        return text
    return text[:max_len-3] + "..."


class LaTeXExporter:
    """Export ablation results to LaTeX format."""
    
    # Default paths for metadata
    DEFAULT_PAPER_MAPPING = Path('data/raw/academic/scopus_2023/paper_mapping.json')
    DLA_PIPER_BASE_URL = 'https://intelligence.dlapiper.com/artificial-intelligence/?t=01-law&c='
    
    def __init__(self, results_json: Path, is_detailed: bool = False, 
                 paper_mapping_path: Path = None):
        """
        Load results from JSON file.
        
        Args:
            results_json: Path to JSON results file
            is_detailed: Whether this is a detailed run (8 queries with full data)
            paper_mapping_path: Path to paper_mapping.json (default: data/processed/paper_mapping.json)
        """
        with open(results_json) as f:
            self.results = json.load(f)
        
        self.is_detailed = is_detailed
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Load paper mapping for citations
        paper_mapping_path = paper_mapping_path or self.DEFAULT_PAPER_MAPPING
        self.paper_mapping = {}
        if paper_mapping_path.exists():
            with open(paper_mapping_path) as f:
                self.paper_mapping = json.load(f)
            print(f"Loaded {len(self.paper_mapping)} paper citations from {paper_mapping_path}")
        else:
            print(f"Warning: Paper mapping not found at {paper_mapping_path}")
        
        # Group by query
        self.by_query = {}
        for r in self.results:
            q = r['query']
            if q not in self.by_query:
                self.by_query[q] = {}
            self.by_query[q][r['mode']] = r
        
        # Group by mode
        self.by_mode = {'semantic': [], 'graph': [], 'dual': []}
        for r in self.results:
            self.by_mode[r['mode']].append(r)
        
        # Group by category
        self.by_category = {}
        for r in self.results:
            cat = r.get('category', 'unknown')
            if cat not in self.by_category:
                self.by_category[cat] = {}
            q = r['query']
            if q not in self.by_category[cat]:
                self.by_category[cat][q] = {}
            self.by_category[cat][q][r['mode']] = r
        
        # Check if detailed data available
        sample = self.results[0] if self.results else {}
        self.has_answer = 'answer_text' in sample and sample.get('answer_text')
        self.has_chunks = 'chunks_detail' in sample and sample.get('chunks_detail')
    
    def format_citation(self, doc_id: str, doc_type: str, jurisdiction: str = None) -> tuple:
        """
        Format a proper citation for a document.
        
        Returns:
            (citation_text, url) tuple
        """
        if doc_type == 'paper':
            # Look up in paper_mapping
            if doc_id in self.paper_mapping:
                meta = self.paper_mapping[doc_id].get('scopus_metadata', {})
                authors = meta.get('authors', 'Unknown')
                # Truncate long author lists
                if authors.count(';') > 2:
                    first_author = authors.split(';')[0].strip()
                    authors = f"{first_author} et al."
                year = meta.get('year', 'n.d.')
                title = meta.get('title', 'Untitled')
                # Truncate long titles
                if len(title) > 80:
                    title = title[:77] + '...'
                journal = meta.get('journal', '')
                doi = meta.get('doi', '')
                link = meta.get('link', '')
                
                citation = f"{authors} ({year}). ``{title}'' {journal}."
                url = f"https://doi.org/{doi}" if doi else link
                return (citation, url)
            else:
                return (f"Paper: {doc_id}", "")
        
        elif doc_type == 'regulation':
            # DLA Piper format
            jur = jurisdiction or doc_id
            citation = f"DLA Piper AI Laws -- {jur}"
            url = f"{self.DLA_PIPER_BASE_URL}{jur}#insight"
            return (citation, url)
        
        else:
            return (f"Source: {doc_id}", "")
    
    def generate_vars_tex(self) -> str:
        """Generate LaTeX variable definitions for inline citations."""
        lines = [
            "% Auto-generated ablation study variables",
            f"% Generated: {datetime.now().isoformat()}",
            f"% Queries: {len(self.by_query)}",
            "",
            "% === MODE AVERAGES ===",
        ]
        
        for mode in ['semantic', 'graph', 'dual']:
            results = self.by_mode[mode]
            if results:
                faith_avg = sum(r['ragas']['faithfulness_score'] for r in results) / len(results)
                rel_avg = sum(r['ragas']['relevancy_score'] for r in results) / len(results)
                
                mode_cap = mode.capitalize()
                lines.append(f"\\newcommand{{\\faith{mode_cap}}}{{{faith_avg:.2f}}}")
                lines.append(f"\\newcommand{{\\relev{mode_cap}}}{{{rel_avg:.2f}}}")
        
        # Mode comparison (which mode wins most)
        winners = {'semantic': 0, 'graph': 0, 'dual': 0}
        for query, modes in self.by_query.items():
            scores = {m: modes[m]['ragas']['faithfulness_score'] for m in modes}
            if scores:
                best = max(scores, key=scores.get)
                winners[best] += 1
        
        lines.extend([
            "",
            "% === WINNER COUNTS ===",
            f"\\newcommand{{\\winnersSemantic}}{{{winners['semantic']}}}",
            f"\\newcommand{{\\winnersGraph}}{{{winners['graph']}}}",
            f"\\newcommand{{\\winnersDual}}{{{winners['dual']}}}",
        ])
        
        # Category averages
        lines.extend([
            "",
            "% === CATEGORY AVERAGES ===",
        ])
        
        for cat, queries in self.by_category.items():
            cat_safe = cat.replace('_', '')
            for mode in ['semantic', 'graph', 'dual']:
                scores = []
                for q, modes in queries.items():
                    if mode in modes:
                        scores.append(modes[mode]['ragas']['faithfulness_score'])
                if scores:
                    avg = sum(scores) / len(scores)
                    mode_cap = mode.capitalize()
                    lines.append(f"\\newcommand{{\\faith{cat_safe}{mode_cap}}}{{{avg:.2f}}}")
        
        return "\n".join(lines)
    
    def generate_grouped_table_tex(self) -> str:
        """Generate results table GROUPED BY CATEGORY (for full/stats mode)."""
        lines = [
            "% Auto-generated ablation results table (grouped by category)",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Ablation Results by Category}",
            "\\label{tab:ablation_results}",
            "\\resizebox{\\linewidth}{!}{%",
            "\\begin{tabular}{llccccccc}",
            "\\toprule",
            "Category & Query & Mode & Faith. & Relev. & Subgraph & S/R/E & Sim. & P/R \\\\",
            "\\midrule",
        ]
        
        for cat, queries in sorted(self.by_category.items()):
            # Category header
            cat_escaped = escape_latex(cat.replace('_', ' '))
            lines.append(f"\\multicolumn{{9}}{{l}}{{\\textbf{{{cat_escaped}}}}} \\\\")
            lines.append("\\midrule")
            
            for query, modes in queries.items():
                q_escaped = escape_latex(truncate_text(query, 40))
                
                for i, mode in enumerate(['semantic', 'graph', 'dual']):
                    if mode not in modes:
                        continue
                    
                    r = modes[mode]
                    
                    # Find best faithfulness for this query
                    all_faith = [modes[m]['ragas']['faithfulness_score'] for m in modes]
                    max_faith = max(all_faith)
                    is_best = r['ragas']['faithfulness_score'] == max_faith and max_faith > 0
                    
                    # Format values
                    faith = r['ragas']['faithfulness_score']
                    faith_str = f"\\textbf{{{faith:.2f}}}" if is_best else f"{faith:.2f}"
                    relev = r['ragas']['relevancy_score']
                    
                    # Subgraph
                    sg_ent = r['graph_utilization']['entities_in_subgraph']
                    sg_rel = r['graph_utilization']['relations_in_subgraph']
                    subgraph = f"{sg_ent}/{sg_rel}"
                    
                    # Chunks by source
                    src = r['retrieval']['chunks_by_source']
                    s = src.get('semantic', 0)
                    rel_c = src.get('graph_provenance', 0)
                    e = src.get('graph_entity', 0)
                    sre = f"{s}/{rel_c}/{e}"
                    
                    # Similarity
                    sim = r['retrieval'].get('avg_query_similarity', 0)
                    
                    # Source diversity
                    src_div = r['retrieval'].get('source_diversity', {})
                    p_count = src_div.get('paper', 0)
                    r_count = src_div.get('regulation', 0)
                    pr = f"{p_count}/{r_count}"
                    
                    # Row content
                    if i == 0:
                        lines.append(f" & {q_escaped} & {mode} & {faith_str} & {relev:.2f} & {subgraph} & {sre} & {sim:.2f} & {pr} \\\\")
                    else:
                        lines.append(f" & & {mode} & {faith_str} & {relev:.2f} & {subgraph} & {sre} & {sim:.2f} & {pr} \\\\")
                
                lines.append("\\cmidrule{2-9}")
            
            lines.append("\\midrule")
        
        # Remove last midrule and add bottomrule
        while lines[-1] in ["\\midrule", "\\cmidrule{2-9}"]:
            lines.pop()
        lines.append("\\bottomrule")
        
        lines.extend([
            "\\end{tabular}}",
            "\\begin{minipage}{\\linewidth}",
            "\\vspace{2mm}",
            "\\footnotesize\\textit{Legend:} "
            "Faith.\\ = Faithfulness; "
            "Relev.\\ = Relevancy; "
            "Subgraph = nodes/relations; "
            "S/R/E = semantic/relation/entity chunks; "
            "Sim.\\ = avg similarity; "
            "P/R = paper/regulation sources. "
            "\\textbf{Bold} = best per query.",
            "\\end{minipage}",
            "\\end{table}",
        ])
        
        return "\n".join(lines)
    
    def generate_flat_table_tex(self) -> str:
        """Generate results table as FLAT LIST (for detailed mode)."""
        lines = [
            "% Auto-generated ablation results table (flat list)",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Detailed Ablation Results}",
            "\\label{tab:ablation_detailed}",
            "\\resizebox{\\linewidth}{!}{%",
            "\\begin{tabular}{clccccccc}",
            "\\toprule",
            "\\# & Query & Mode & Faith. & Relev. & Subgraph & S/R/E & Sim. & P/R \\\\",
            "\\midrule",
        ]
        
        row_num = 1
        for query, modes in self.by_query.items():
            q_escaped = escape_latex(truncate_text(query, 45))
            
            for i, mode in enumerate(['semantic', 'graph', 'dual']):
                if mode not in modes:
                    continue
                
                r = modes[mode]
                
                # Find best faithfulness for this query
                all_faith = [modes[m]['ragas']['faithfulness_score'] for m in modes]
                max_faith = max(all_faith)
                is_best = r['ragas']['faithfulness_score'] == max_faith and max_faith > 0
                
                # Format values
                faith = r['ragas']['faithfulness_score']
                faith_str = f"\\textbf{{{faith:.2f}}}" if is_best else f"{faith:.2f}"
                relev = r['ragas']['relevancy_score']
                
                # Subgraph
                sg_ent = r['graph_utilization']['entities_in_subgraph']
                sg_rel = r['graph_utilization']['relations_in_subgraph']
                subgraph = f"{sg_ent}/{sg_rel}"
                
                # Chunks by source
                src = r['retrieval']['chunks_by_source']
                s = src.get('semantic', 0)
                rel_c = src.get('graph_provenance', 0)
                e = src.get('graph_entity', 0)
                sre = f"{s}/{rel_c}/{e}"
                
                # Similarity
                sim = r['retrieval'].get('avg_query_similarity', 0)
                
                # Source diversity
                src_div = r['retrieval'].get('source_diversity', {})
                p_count = src_div.get('paper', 0)
                r_count = src_div.get('regulation', 0)
                pr = f"{p_count}/{r_count}"
                
                # Row content
                if i == 0:
                    lines.append(f"{row_num} & {q_escaped} & {mode} & {faith_str} & {relev:.2f} & {subgraph} & {sre} & {sim:.2f} & {pr} \\\\")
                else:
                    lines.append(f" & & {mode} & {faith_str} & {relev:.2f} & {subgraph} & {sre} & {sim:.2f} & {pr} \\\\")
            
            lines.append("\\midrule")
            row_num += 1
        
        # Remove last midrule and add bottomrule
        if lines[-1] == "\\midrule":
            lines.pop()
        lines.append("\\bottomrule")
        
        lines.extend([
            "\\end{tabular}}",
            "\\begin{minipage}{\\linewidth}",
            "\\vspace{2mm}",
            "\\footnotesize\\textit{Legend:} "
            "Faith.\\ = Faithfulness; "
            "Relev.\\ = Relevancy; "
            "Subgraph = nodes/relations; "
            "S/R/E = semantic/relation/entity chunks; "
            "Sim.\\ = avg similarity; "
            "P/R = paper/regulation sources. "
            "\\textbf{Bold} = best per query.",
            "\\end{minipage}",
            "\\end{table}",
        ])
        
        return "\n".join(lines)
    
    def generate_appendix_tex(self) -> str:
        """
        Generate appendix with full query details.
        
        ONLY for detailed mode - returns minimal stub for stats mode.
        """
        if not self.is_detailed:
            # Stats mode: no appendix, just a comment
            return "% No appendix for stats mode. Use --detailed for full appendix.\n"
        
        lines = [
            "% Auto-generated detailed ablation appendix",
            "% DETAILED MODE: includes full answers, chunks, and relations",
            "\\chapter{Detailed Query Analysis}",
            "\\label{app:detailed_queries}",
            "",
            f"This appendix presents the complete analysis for {len(self.by_query)} representative queries.",
            "Each section includes metrics comparison, full generated answers,",
            "retrieved chunks with source provenance, and extracted relations.",
            "",
        ]
        
        for i, (query, modes) in enumerate(self.by_query.items(), 1):
            q_escaped = escape_latex(query)
            
            # Get query metadata from first result
            first_result = list(modes.values())[0]
            category = first_result.get('category', 'unknown')
            subcategory = first_result.get('subcategory', '')
            
            lines.extend([
                f"\\section{{Query {i}: {q_escaped}}}",
                f"\\label{{sec:q{i}}}",
                "",
                f"\\textbf{{Category:}} {escape_latex(category)}" + (f" ({escape_latex(subcategory)})" if subcategory else ""),
                "",
            ])
            
            # ─────────────────────────────────────────────────────────────────
            # SECTION 1: Summary Statistics Box
            # ─────────────────────────────────────────────────────────────────
            lines.extend([
                "\\subsection*{Summary Statistics}",
                "\\begin{tcolorbox}[colback=blue!5,colframe=blue!50,title=Mode Comparison]",
            ])
            
            # Find winner
            scores = {m: modes[m]['ragas']['faithfulness_score'] for m in modes}
            winner = max(scores, key=scores.get) if scores else 'N/A'
            winner_score = scores.get(winner, 0)
            
            lines.append(f"\\textbf{{Winner:}} {winner.upper()} (Faithfulness: {winner_score:.2f}) \\\\")
            
            # Mode summary line
            for mode in ['semantic', 'graph', 'dual']:
                if mode in modes:
                    r = modes[mode]
                    f_score = r['ragas']['faithfulness_score']
                    chunks = r['retrieval']['total_chunks']
                    sg = r['graph_utilization']['entities_in_subgraph']
                    lines.append(f"\\textit{{{mode.capitalize()}}}: Faith={f_score:.2f}, Chunks={chunks}, Subgraph={sg} nodes \\\\")
            
            lines.extend([
                "\\end{tcolorbox}",
                "",
            ])
            
            # ─────────────────────────────────────────────────────────────────
            # SECTION 2: Metrics Comparison Table
            # ─────────────────────────────────────────────────────────────────
            lines.extend([
                "\\subsection*{Detailed Metrics}",
                "\\begin{table}[H]",
                "\\centering",
                "\\begin{tabular}{lccc}",
                "\\toprule",
                "Metric & Semantic & Graph & Dual \\\\",
                "\\midrule",
            ])
            
            metrics = [
                ('Faithfulness', lambda r: f"{r['ragas']['faithfulness_score']:.2f}"),
                ('Relevancy', lambda r: f"{r['ragas']['relevancy_score']:.2f}"),
                ('Entities Resolved', lambda r: str(r['entity_resolution']['resolved_count'])),
                ('Subgraph Nodes', lambda r: str(r['graph_utilization']['entities_in_subgraph'])),
                ('Subgraph Relations', lambda r: str(r['graph_utilization']['relations_in_subgraph'])),
                ('Total Chunks', lambda r: str(r['retrieval']['total_chunks'])),
                ('Avg Similarity', lambda r: f"{r['retrieval'].get('avg_query_similarity', 0):.2f}"),
                ('Terminal Coverage', lambda r: f"{r['coverage'].get('terminal_coverage_rate', 0)*100:.0f}\\%"),
            ]
            
            for metric_name, metric_fn in metrics:
                vals = []
                for mode in ['semantic', 'graph', 'dual']:
                    if mode in modes:
                        vals.append(metric_fn(modes[mode]))
                    else:
                        vals.append("---")
                lines.append(f"{metric_name} & {vals[0]} & {vals[1]} & {vals[2]} \\\\")
            
            lines.extend([
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
                "",
            ])
            
            # ─────────────────────────────────────────────────────────────────
            # SECTION 3: Full Answers (for each mode)
            # ─────────────────────────────────────────────────────────────────
            if self.has_answer:
                lines.append("\\subsection*{Generated Answers}")
                
                for mode in ['semantic', 'graph', 'dual']:
                    if mode not in modes:
                        continue
                    
                    r = modes[mode]
                    answer = r.get('answer_text', '')
                    
                    if not answer:
                        answer = "(No answer recorded)"
                    
                    # FULL answer text - no truncation
                    answer_escaped = escape_latex(answer)
                    faith = r['ragas']['faithfulness_score']
                    relev = r['ragas']['relevancy_score']
                    
                    lines.extend([
                        f"\\subsubsection*{{{mode.capitalize()} Mode}}",
                        f"\\textit{{Faithfulness: {faith:.2f}, Relevancy: {relev:.2f}}}",
                        "",
                        "\\begin{tcolorbox}[colback=gray!5,colframe=gray!50,boxrule=0.5pt,breakable]",
                        f"\\small {answer_escaped}",
                        "\\end{tcolorbox}",
                        "",
                    ])
            
            # ─────────────────────────────────────────────────────────────────
            # SECTION 4: Retrieved Chunks (FULL TEXT for each mode)
            # ─────────────────────────────────────────────────────────────────
            if self.has_chunks:
                lines.append("\\subsection*{Retrieved Chunks}")
                
                for mode in ['semantic', 'graph', 'dual']:
                    if mode not in modes:
                        continue
                    
                    r = modes[mode]
                    chunks = r.get('chunks_detail', [])
                    cited = r.get('cited_chunks', [])
                    
                    if not chunks:
                        continue
                    
                    lines.extend([
                        f"\\subsubsection*{{{mode.capitalize()} Mode Chunks}}",
                        "",
                    ])
                    
                    for c in chunks:  # ALL chunks, FULL text
                        idx = c.get('index', '?')
                        doc_id = c.get('doc_id', 'unknown')
                        doc_type = c.get('doc_type', 'unknown')
                        score = c.get('score', 0)
                        jurisdiction = c.get('jurisdiction', None)
                        method = c.get('method', 'unknown')
                        is_cited = c.get('cited', False) or idx in cited
                        
                        # FULL text - no truncation
                        text = escape_latex(c.get('text', ''))
                        
                        # Format proper citation
                        citation, url = self.format_citation(doc_id, doc_type, jurisdiction)
                        citation_escaped = escape_latex(citation)
                        
                        cited_str = " $\\checkmark$ \\textbf{CITED}" if is_cited else ""
                        
                        lines.append(f"\\paragraph{{Chunk {idx}{cited_str}}}")
                        lines.append(f"\\textit{{{citation_escaped}}}")
                        if url:
                            lines.append(f"\\\\\\url{{{url}}}")
                        lines.append(f"\\\\\\textit{{Score: {score:.3f} | Method: {method}}}")
                        lines.extend([
                            "",
                            "\\begin{quote}",
                            f"\\small {text}",
                            "\\end{quote}",
                            "",
                        ])
                    
                    lines.append("")
            
            # ─────────────────────────────────────────────────────────────────
            # SECTION 5: Relations (FULL for graph/dual modes)
            # ─────────────────────────────────────────────────────────────────
            lines.append("\\subsection*{Extracted Relations}")
            
            for mode in ['graph', 'dual']:  # Only modes with relations
                if mode not in modes:
                    continue
                
                r = modes[mode]
                relations = r.get('relations_detail', [])
                
                if not relations:
                    lines.append(f"\\textit{{No relations extracted for {mode} mode.}}")
                    lines.append("")
                    continue
                
                lines.extend([
                    f"\\subsubsection*{{{mode.capitalize()} Mode Relations ({len(relations)} total)}}",
                    "\\begin{longtable}{p{4cm}lp{4cm}p{3cm}}",
                    "\\toprule",
                    "Subject & Predicate & Object & Provenance \\\\",
                    "\\midrule",
                    "\\endhead",
                ])
                
                for rel in relations:  # ALL relations, FULL text
                    subj = escape_latex(rel.get('subject_id', ''))
                    pred = escape_latex(rel.get('predicate', ''))
                    obj = escape_latex(rel.get('object_id', ''))
                    chunk_ids = rel.get('chunk_ids', [])
                    provenance = ', '.join(chunk_ids[:3]) if chunk_ids else 'N/A'
                    provenance = escape_latex(provenance)
                    
                    lines.append(f"{subj} & {pred} & {obj} & {provenance} \\\\")
                
                lines.extend([
                    "\\bottomrule",
                    "\\end{longtable}",
                    "",
                ])
            
            lines.append("\\clearpage")
            lines.append("")
        
        return "\n".join(lines)
    
    def generate_pgfplots_data(self) -> str:
        """Generate data tables for pgfplots."""
        lines = [
            "% Auto-generated data for pgfplots",
            f"% Generated: {datetime.now().isoformat()}",
            "",
        ]
        
        # Category-mode faithfulness table
        lines.extend([
            "% Data for category × mode bar chart",
            "\\pgfplotstableread{",
            "category semantic graph dual",
        ])
        
        for cat, queries in sorted(self.by_category.items()):
            cat_safe = cat.replace('_', '-')
            mode_avgs = {}
            for mode in ['semantic', 'graph', 'dual']:
                scores = []
                for q, modes in queries.items():
                    if mode in modes:
                        scores.append(modes[mode]['ragas']['faithfulness_score'])
                mode_avgs[mode] = sum(scores) / len(scores) if scores else 0
            
            lines.append(f"{cat_safe} {mode_avgs['semantic']:.3f} {mode_avgs['graph']:.3f} {mode_avgs['dual']:.3f}")
        
        lines.append("}\\categorytable")
        lines.append("")
        
        # Mode summary table
        lines.extend([
            "% Data for mode summary",
            "\\pgfplotstableread{",
            "mode faithfulness relevancy",
        ])
        
        for mode in ['semantic', 'graph', 'dual']:
            results = self.by_mode[mode]
            if results:
                faith_avg = sum(r['ragas']['faithfulness_score'] for r in results) / len(results)
                rel_avg = sum(r['ragas']['relevancy_score'] for r in results) / len(results)
                lines.append(f"{mode} {faith_avg:.3f} {rel_avg:.3f}")
        
        lines.append("}\\modetable")
        
        return "\n".join(lines)
    
    def export_all(self, output_dir: Path):
        """
        Export LaTeX files based on mode.
        
        Detailed mode (8 queries):
          - ablation_vars.tex (macros for inline use)
          - ablation_table.tex (flat list, no grouping)
          - ablation_appendix.tex (full I/O per query)
        
        Stats/Full mode (36 queries):
          - ablation_table.tex (grouped by category)
          - ablation_data.tex (pgfplots data)
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        files = {}
        
        if self.is_detailed:
            # DETAILED MODE: vars, flat table, appendix
            
            # Variables (macros for inline citations)
            vars_file = output_dir / f'ablation_vars_{self.timestamp}.tex'
            with open(vars_file, 'w') as f:
                f.write(self.generate_vars_tex())
            print(f"Variables: {vars_file}")
            files['vars'] = vars_file
            
            # Table (flat list, no category grouping)
            table_file = output_dir / f'ablation_table_{self.timestamp}.tex'
            with open(table_file, 'w') as f:
                f.write(self.generate_flat_table_tex())
            print(f"Table: {table_file}")
            files['table'] = table_file
            
            # Appendix (full I/O)
            appendix_file = output_dir / f'ablation_appendix_{self.timestamp}.tex'
            with open(appendix_file, 'w') as f:
                f.write(self.generate_appendix_tex())
            print(f"Appendix: {appendix_file}")
            files['appendix'] = appendix_file
            
        else:
            # STATS/FULL MODE: grouped table, pgfplots data
            
            # Table (grouped by category)
            table_file = output_dir / f'ablation_table_{self.timestamp}.tex'
            with open(table_file, 'w') as f:
                f.write(self.generate_grouped_table_tex())
            print(f"Table: {table_file}")
            files['table'] = table_file
            
            # Data for plots
            data_file = output_dir / f'ablation_data_{self.timestamp}.tex'
            with open(data_file, 'w') as f:
                f.write(self.generate_pgfplots_data())
            print(f"Plot data: {data_file}")
            files['data'] = data_file
        
        return files


def main():
    """CLI for LaTeX export."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Export ablation results to LaTeX')
    parser.add_argument('json_file', type=Path, help='Input JSON results file')
    parser.add_argument('-o', '--output', type=Path, default=Path('data/analysis/latex'),
                        help='Output directory')
    parser.add_argument('--detailed', action='store_true',
                        help='Force detailed mode (full appendix with answers/chunks)')
    parser.add_argument('--paper-mapping', type=Path, default=None,
                        help='Path to paper_mapping.json (default: data/processed/paper_mapping.json)')
    
    args = parser.parse_args()
    
    if not args.json_file.exists():
        print(f"Error: {args.json_file} not found")
        return 1
    
    # Auto-detect detailed mode from filename
    is_detailed = args.detailed or 'detailed' in args.json_file.name
    
    exporter = LaTeXExporter(
        args.json_file, 
        is_detailed=is_detailed,
        paper_mapping_path=args.paper_mapping
    )
    exporter.export_all(args.output)
    
    print(f"\nLaTeX export complete! (detailed={is_detailed})")
    return 0


if __name__ == '__main__':
    exit(main())