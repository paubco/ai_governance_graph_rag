# -*- coding: utf-8 -*-
"""
Document

Loads documents from different sources (DLA Piper regulations and academic papers)
into standardized Document format for downstream processing. Handles regulations
(one document per country, concatenated sections) and academic papers (markdown-
preserved) with detailed provenance metadata tracking.

Examples:
loader = DocumentLoader(year='2023')
    documents = loader.load_all_documents()
    # Returns: List[Document] with regulations and papers

"""
"""
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class Document:
    """
    Standardized document representation.
    
    Attributes:
        doc_id: Unique identifier (e.g., "reg_EU", "paper_001")
        source_type: Either "regulation" or "academic_paper"
        title: Human-readable title
        text: Full text content
        metadata: Source-specific metadata dict
    """
    doc_id: str
    source_type: str
    title: str
    text: str
    metadata: Dict
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def __repr__(self) -> str:
        return f"Document(id={self.doc_id}, type={self.source_type}, title='{self.title[:50]}...')"


class DocumentLoader:
    """
    Unified document loader for regulations and academic papers.
    
    Design Decisions:
    1. Regulations: One document per country (all sections concatenated)
    2. Academic Papers: Keep markdown formatting for semantic structure
    3. Metadata: Store detailed metadata for provenance tracking
    
    Usage:
        # Default (uses 2023 data)
        loader = DocumentLoader()
        
        # Specify year
        loader = DocumentLoader(year='2024')
        
        # Load all documents
        documents = loader.load_all_documents()
        
        # Or load separately
        regulations = loader.load_all_regulations()
        papers = loader.load_all_papers()
    """

    def __init__(
        self,
        year: str = '2023',
        regulations_dir: str = 'data/raw/dlapiper',
        academic_base: str = 'data/raw/academic'
    ):
        """
        Initialize the document loader.
        
        Args:
            year: Scopus dataset year (default: '2023')
            regulations_dir: Directory containing DLA Piper JSON files
            academic_base: Base directory for academic data
        """
        self.year = year
        self.regulations_dir = Path(regulations_dir)
        self.academic_base = Path(academic_base)
        
        # Year-specific paths
        self.scopus_dir = self.academic_base / f'scopus_{year}'
        self.papers_dir = self.scopus_dir / 'MinerU_parsed_papers'
        self.scopus_csv = self.scopus_dir / f'scopus_export_{year}_raw.csv'
        self.mapping_json = self.scopus_dir / 'paper_mapping.json'
        
        # Cache
        self.scopus_metadata = None
        self.paper_mapping = None
        
        print(f"DocumentLoader initialized (year={year}):")
        print(f"  Regulations: {self.regulations_dir}")
        print(f"  Papers: {self.papers_dir}")
        print(f"  Scopus CSV: {self.scopus_csv}")
        print(f"  Mapping: {self.mapping_json}")
    
    # ==================== REGULATION LOADING ====================
    
    def load_regulation(self, json_path: Path) -> Document:
        """
        Load a single regulation from DLA Piper JSON.
        
        Args:
            json_path: Path to regulation JSON file
            
        Returns:
            Document object with standardized format
            
        Example input JSON structure:
            {
              "country_code": "EU",
              "country_name": "European Union",
              "url": "https://...",
              "scraped_date": "2025-11-06T16:49:23.957594",
              "num_sections": 11,
              "sections": [
                {
                  "title": "Law / proposed law",
                  "heading": "Law / proposed law in the European Union",
                  "main_content": "...",
                  "country_specific_notes": [],
                  "subsections": [],
                  "links": [{"text": "...", "url": "..."}]
                }
              ]
            }
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create document ID: reg_{country_code}
        doc_id = f"reg_{data['country_code']}"
        
        # Assemble text from all sections
        # Format: Section with clear boundaries for semantic chunking
        text_parts = []
        section_titles = []
        
        for section in data['sections']:
            section_title = section['title']
            section_heading = section.get('heading', '')
            section_content = section['main_content']
            
            section_titles.append(section_title)
            
            # Format section with clear markdown structure
            text_parts.append(f"# {section_title}\n")
            if section_heading and section_heading != section_title:
                text_parts.append(f"## {section_heading}\n")
            text_parts.append(f"\n{section_content}\n")
            text_parts.append("\n---\n\n")  # Section separator
        
        # Join all sections into full text
        full_text = "".join(text_parts)
        
        # Create title (use country name)
        title = f"AI Regulations - {data['country_name']}"
        
        # Prepare metadata
        metadata = {
            'country_code': data['country_code'],
            'country_name': data['country_name'],
            'url': data['url'],
            'scraped_date': data['scraped_date'],
            'num_sections': data['num_sections'],
            'section_titles': section_titles,
            'source': 'DLA Piper AI Laws of the World'
        }
        
        return Document(
            doc_id=doc_id,
            source_type='regulation',
            title=title,
            text=full_text,
            metadata=metadata
        )
    
    def load_all_regulations(self) -> List[Document]:
        """
        Load all regulations from the regulations directory.
        
        Returns:
            List of Document objects, one per country
        """
        if not self.regulations_dir.exists():
            raise FileNotFoundError(
                f"Regulations directory not found: {self.regulations_dir}"
            )
        
        # Find all JSON files
        json_files = list(self.regulations_dir.glob("*.json"))
        
        if not json_files:
            raise FileNotFoundError(
                f"No JSON files found in {self.regulations_dir}"
            )
        
        print(f"\nLoading {len(json_files)} regulations...")
        
        documents = []
        for json_path in sorted(json_files):
            try:
                doc = self.load_regulation(json_path)
                documents.append(doc)
                print(f"  ✓ Loaded: {doc.doc_id} ({doc.metadata['country_name']})")
            except Exception as e:
                print(f"  ✗ Failed to load {json_path.name}: {e}")
        
        print(f"\nSuccessfully loaded {len(documents)}/{len(json_files)} regulations")
        return documents
    
    # ==================== ACADEMIC PAPER LOADING ====================
    
    def _load_scopus_metadata(self) -> Dict[str, Dict]:
        """
        Load Scopus metadata CSV into memory.
        
        Returns:
            Dictionary keyed by DOI for fast lookup
        """
        if self.scopus_metadata is not None:
            return self.scopus_metadata
        
        if not self.scopus_csv.exists():
            raise FileNotFoundError(
                f"Scopus CSV not found: {self.scopus_csv}"
            )
        
        print(f"\nLoading Scopus metadata from {self.scopus_csv.name}...")
        
        metadata = {}
        with open(self.scopus_csv, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                doi = row.get('DOI', '').strip()
                if doi:
                    metadata[doi] = row
        
        print(f"  Loaded metadata for {len(metadata)} papers")
        self.scopus_metadata = metadata
        return metadata
    
    def _load_paper_mapping(self) -> Dict[str, Dict]:
        """
        Load paper mapping JSON (paper_XXX → Scopus metadata).
        
        Returns:
            Dictionary with paper IDs as keys
        """
        if self.paper_mapping is not None:
            return self.paper_mapping
        
        if not self.mapping_json.exists():
            raise FileNotFoundError(
                f"Paper mapping not found: {self.mapping_json}"
            )
        
        print(f"\nLoading paper mapping from {self.mapping_json.name}...")
        
        with open(self.mapping_json, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        
        print(f"  Loaded mappings for {len(mapping)} papers")
        self.paper_mapping = mapping
        return mapping
    
    def load_academic_paper(self, paper_folder: Path) -> Optional[Document]:
        """
        Load a single academic paper from MinerU output + Scopus metadata.
        
        Args:
            paper_folder: Path to paper folder (e.g., paper_001/)
            
        Returns:
            Document object or None if loading fails
            
        Paper folder structure:
            paper_001/
                full.md             # MinerU extracted text
                content_list.json   # MinerU metadata
        """
        # Get paper ID from folder name
        paper_id = paper_folder.name  # e.g., "paper_001"
        
        # Load markdown text
        md_path = paper_folder / "full.md"
        if not md_path.exists():
            print(f"  ✗ Missing full.md for {paper_id}")
            return None
        
        with open(md_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Load Scopus metadata and paper mapping if not already loaded
        if self.scopus_metadata is None:
            self._load_scopus_metadata()
        if self.paper_mapping is None:
            self._load_paper_mapping()
        
        # Look up which Scopus entry this paper matches
        if paper_id not in self.paper_mapping:
            print(f"  ✗ No mapping found for {paper_id}")
            return None
        
        paper_info = self.paper_mapping[paper_id]
        
        # Get scopus_metadata from the nested structure
        scopus_metadata = paper_info.get('scopus_metadata', {})
        
        if not scopus_metadata:
            print(f"  ✗ No Scopus metadata for {paper_id}")
            return None
        
        # Extract DOI from nested metadata
        scopus_doi = scopus_metadata.get('doi')
        
        if not scopus_doi:
            print(f"  ✗ No DOI found for {paper_id}")
            return None
        
        # Get full Scopus metadata from CSV (for additional fields)
        scopus_row = self.scopus_metadata.get(scopus_doi, {})
        
        # Use metadata from paper_mapping (which includes match info)
        # But supplement with CSV data if available
        title = scopus_metadata.get('title') or scopus_row.get('Title', 'Untitled Paper')
        authors = scopus_metadata.get('authors') or scopus_row.get('Authors', 'Unknown')
        year = scopus_metadata.get('year') or scopus_row.get('Year', 'Unknown')
        
        # Create document
        doc_id = paper_id
        
        metadata = {
            'doi': scopus_doi,
            'title': title,
            'authors': authors,
            'year': year,
            'journal': scopus_metadata.get('journal') or scopus_row.get('Source title', ''),
            'abstract': scopus_metadata.get('abstract') or scopus_row.get('Abstract', ''),
            'citations': scopus_metadata.get('citations', 0),
            'eid': scopus_metadata.get('eid', ''),
            'match_method': scopus_metadata.get('match_method', 'unknown'),
            'match_confidence': scopus_metadata.get('match_confidence', 0.0),
            'affiliations': scopus_row.get('Affiliations', ''),
            'author_keywords': scopus_row.get('Author Keywords', ''),
            'index_keywords': scopus_row.get('Index Keywords', ''),
            'link': scopus_row.get('Link', ''),
            'source': 'Scopus + MinerU'
        }
        
        return Document(
            doc_id=doc_id,
            source_type='academic_paper',
            title=title,
            text=text,
            metadata=metadata
        )
    
    def load_all_papers(self) -> List[Document]:
        """
        Load all academic papers from the papers directory.
        
        Returns:
            List of Document objects, one per paper
        """
        if not self.papers_dir.exists():
            raise FileNotFoundError(
                f"Papers directory not found: {self.papers_dir}"
            )
        
        # Find all paper folders (paper_001, paper_002, etc.)
        paper_folders = [
            p for p in self.papers_dir.iterdir() 
            if p.is_dir() and p.name.startswith('paper_')
        ]
        
        if not paper_folders:
            raise FileNotFoundError(
                f"No paper folders found in {self.papers_dir}"
            )
        
        print(f"\nLoading {len(paper_folders)} academic papers...")
        
        documents = []
        for paper_folder in sorted(paper_folders):
            try:
                doc = self.load_academic_paper(paper_folder)
                if doc:
                    documents.append(doc)
                    print(f"  ✓ Loaded: {doc.doc_id} - {doc.title[:60]}...")
            except Exception as e:
                print(f"  ✗ Failed to load {paper_folder.name}: {e}")
        
        print(f"\nSuccessfully loaded {len(documents)}/{len(paper_folders)} papers")
        return documents
    
    # ==================== UNIFIED LOADING ====================
    
    def load_all_documents(self) -> List[Document]:
        """
        Load all documents (regulations + academic papers).
        
        Returns:
            Combined list of all Document objects
        """
        print("="*60)
        print("LOADING ALL DOCUMENTS")
        print("="*60)
        
        regulations = self.load_all_regulations()
        papers = self.load_all_papers()
        
        all_docs = regulations + papers
        
        print("\n" + "="*60)
        print(f"TOTAL DOCUMENTS LOADED: {len(all_docs)}")
        print(f"  Regulations: {len(regulations)}")
        print(f"  Academic Papers: {len(papers)}")
        print("="*60)
        
        return all_docs
    
    # ==================== UTILITIES ====================
    
    def save_documents(self, documents: List[Document], output_path: str):
        """
        Save loaded documents to JSON file.
        
        Args:
            documents: List of Document objects
            output_path: Path to save JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict format
        docs_dict = [doc.to_dict() for doc in documents]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(docs_dict, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Saved {len(documents)} documents to {output_path}")
    
    def get_stats(self, documents: List[Document]) -> Dict:
        """
        Get statistics about loaded documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_documents': len(documents),
            'by_type': {},
            'total_chars': sum(len(doc.text) for doc in documents),
            'avg_chars': 0,
            'regulations': {},
            'papers': {}
        }
        
        # Count by type
        for doc in documents:
            doc_type = doc.source_type
            stats['by_type'][doc_type] = stats['by_type'].get(doc_type, 0) + 1
        
        # Average characters
        if documents:
            stats['avg_chars'] = stats['total_chars'] // len(documents)
        
        # Regulation stats
        regs = [d for d in documents if d.source_type == 'regulation']
        if regs:
            stats['regulations']['count'] = len(regs)
            stats['regulations']['avg_sections'] = sum(
                d.metadata['num_sections'] for d in regs
            ) / len(regs)
        
        # Paper stats
        papers = [d for d in documents if d.source_type == 'academic_paper']
        if papers:
            stats['papers']['count'] = len(papers)
            stats['papers']['years'] = list(set(
                str(d.metadata['year']) for d in papers
            ))
        
        return stats


# ==================== TESTING & DEMO ====================

def main():
    """
    Demo script to test the document loader.
    
    Run with:
        python src/ingestion/document_loader.py
    """
    # Initialize loader (default 2023)
    loader = DocumentLoader()
    
    # Load all documents
    documents = loader.load_all_documents()
    
    # Print statistics
    print("\n" + "="*60)
    print("DOCUMENT STATISTICS")
    print("="*60)
    stats = loader.get_stats(documents)
    print(f"Total Documents: {stats['total_documents']}")
    print(f"  Regulations: {stats['by_type'].get('regulation', 0)}")
    print(f"  Academic Papers: {stats['by_type'].get('academic_paper', 0)}")
    print(f"\nTotal Characters: {stats['total_chars']:,}")
    print(f"Average Characters per Document: {stats['avg_chars']:,}")
    
    if stats['regulations']:
        print(f"\nRegulations:")
        print(f"  Count: {stats['regulations']['count']}")
        print(f"  Avg Sections: {stats['regulations']['avg_sections']:.1f}")
    
    if stats['papers']:
        print(f"\nAcademic Papers:")
        print(f"  Count: {stats['papers']['count']}")
        print(f"  Years: {', '.join(sorted(stats['papers']['years']))}")
    
    # Save to JSON
    loader.save_documents(
        documents, 
        'data/interim/loaded_documents.json'
    )
    
    # Show sample documents
    print("\n" + "="*60)
    print("SAMPLE DOCUMENTS")
    print("="*60)
    
    if documents:
        print(f"\n[1] First Regulation:")
        reg = next((d for d in documents if d.source_type == 'regulation'), None)
        if reg:
            print(f"  ID: {reg.doc_id}")
            print(f"  Title: {reg.title}")
            print(f"  Text Length: {len(reg.text):,} chars")
            print(f"  Sections: {reg.metadata['num_sections']}")
            print(f"  Preview: {reg.text[:200]}...")
        
        print(f"\n[2] First Academic Paper:")
        paper = next((d for d in documents if d.source_type == 'academic_paper'), None)
        if paper:
            print(f"  ID: {paper.doc_id}")
            print(f"  Title: {paper.title}")
            print(f"  Text Length: {len(paper.text):,} chars")
            print(f"  Authors: {paper.metadata['authors'][:100]}...")
            print(f"  Year: {paper.metadata['year']}")
            print(f"  Preview: {paper.text[:200]}...")


if __name__ == "__main__":
    main()