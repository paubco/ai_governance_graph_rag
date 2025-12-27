# -*- coding: utf-8 -*-
"""
Document

Tests the document loader on a small subset of files to verify functionality
before loading the full dataset. Includes tests for single regulations,
single papers, small batches, and full dataset loading.

"""
"""
import sys
from pathlib import Path

# Add src to path so we can import document_loader
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ingestion.document_loader import DocumentLoader, Document

def test_single_regulation():
    """Test loading a single regulation."""
    print("\n" + "="*60)
    print("TEST 1: Load Single Regulation (EU)")
    print("="*60)
    
    loader = DocumentLoader()
    
    # Try to load EU regulation
    eu_path = Path('data/raw/dlapiper/EU.json')
    
    if not eu_path.exists():
        print(f"✗ File not found: {eu_path}")
        print("  Please check if DLA Piper files are in the correct location")
        return False
    
    try:
        doc = loader.load_regulation(eu_path)
        print(f"✓ Successfully loaded: {doc.doc_id}")
        print(f"  Title: {doc.title}")
        print(f"  Type: {doc.source_type}")
        print(f"  Text length: {len(doc.text):,} characters")
        print(f"  Sections: {doc.metadata['num_sections']}")
        print(f"  Country: {doc.metadata['country_name']}")
        print(f"\n  Text preview (first 300 chars):")
        print(f"  {doc.text[:300]}...")
        return True
    except Exception as e:
        print(f"✗ Error loading regulation: {e}")
        return False


def test_single_paper():
    """Test loading a single academic paper."""
    print("\n" + "="*60)
    print("TEST 2: Load Single Academic Paper")
    print("="*60)
    
    loader = DocumentLoader()
    
    # Find first paper folder
    papers_dir = Path('data/raw/academic/scopus_2023/MinerU_parsed_papers')
    
    if not papers_dir.exists():
        print(f"✗ Directory not found: {papers_dir}")
        return False
    
    paper_folders = sorted([
        p for p in papers_dir.iterdir() 
        if p.is_dir() and p.name.startswith('paper_')
    ])
    
    if not paper_folders:
        print(f"✗ No paper folders found in {papers_dir}")
        return False
    
    first_paper = paper_folders[0]
    print(f"  Testing with: {first_paper.name}")
    
    try:
        doc = loader.load_academic_paper(first_paper)
        
        if doc is None:
            print(f"✗ Failed to load paper (returned None)")
            return False
        
        print(f"✓ Successfully loaded: {doc.doc_id}")
        print(f"  Title: {doc.title}")
        print(f"  Type: {doc.source_type}")
        print(f"  Text length: {len(doc.text):,} characters")
        print(f"  Authors: {doc.metadata['authors'][:80]}...")
        print(f"  Year: {doc.metadata['year']}")
        print(f"  DOI: {doc.metadata['doi']}")
        print(f"\n  Text preview (first 300 chars):")
        print(f"  {doc.text[:300]}...")
        return True
    except Exception as e:
        print(f"✗ Error loading paper: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_load_small_batch():
    """Test loading a small batch (5 regulations, 5 papers)."""
    print("\n" + "="*60)
    print("TEST 3: Load Small Batch (5 of each)")
    print("="*60)
    
    loader = DocumentLoader()
    
    # Load first 5 regulations
    reg_files = sorted(list(Path('data/raw/dlapiper').glob('*.json')))[:5]
    print(f"\nLoading {len(reg_files)} regulations...")
    
    regulations = []
    for reg_file in reg_files:
        try:
            doc = loader.load_regulation(reg_file)
            regulations.append(doc)
            print(f"  ✓ {doc.doc_id}: {doc.metadata['country_name']}")
        except Exception as e:
            print(f"  ✗ {reg_file.name}: {e}")
    
    # Load first 5 papers
    paper_folders = sorted([
        p for p in Path('data/raw/academic/scopus_2023/MinerU_parsed_papers').iterdir()
        if p.is_dir() and p.name.startswith('paper_')
    ])[:5]
    
    print(f"\nLoading {len(paper_folders)} papers...")
    
    papers = []
    for paper_folder in paper_folders:
        try:
            doc = loader.load_academic_paper(paper_folder)
            if doc:
                papers.append(doc)
                print(f"  ✓ {doc.doc_id}: {doc.title[:50]}...")
        except Exception as e:
            print(f"  ✗ {paper_folder.name}: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"BATCH TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Regulations loaded: {len(regulations)}/{len(reg_files)}")
    print(f"Papers loaded: {len(papers)}/{len(paper_folders)}")
    print(f"Total documents: {len(regulations) + len(papers)}")
    
    return len(regulations) > 0 and len(papers) > 0


def test_full_load():
    """Test loading ALL documents."""
    print("\n" + "="*60)
    print("TEST 4: Load All Documents (Full Dataset)")
    print("="*60)
    print("\nThis will load all 48 regulations + ~50 papers...")
    
    response = input("Continue? [y/N]: ").strip().lower()
    if response != 'y':
        print("Skipped.")
        return True
    
    try:
        loader = DocumentLoader()
        documents = loader.load_all_documents()
        
        print(f"\n{'='*60}")
        print(f"FULL LOAD SUCCESS!")
        print(f"{'='*60}")
        
        stats = loader.get_stats(documents)
        print(f"Total documents: {stats['total_documents']}")
        print(f"  Regulations: {stats['by_type'].get('regulation', 0)}")
        print(f"  Papers: {stats['by_type'].get('academic_paper', 0)}")
        print(f"Total text: {stats['total_chars']:,} characters")
        print(f"Average per doc: {stats['avg_chars']:,} characters")
        
        return True
    except Exception as e:
        print(f"✗ Error in full load: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("DOCUMENT LOADER TEST SUITE")
    print("="*60)
    print("\nThis will test the document loader step by step.")
    print("Make sure you're in the Graph_RAG directory!\n")
    
    # Check we're in the right directory
    if not Path('data').exists():
        print("✗ ERROR: 'data' directory not found!")
        print("  Please run this script from the Graph_RAG directory")
        print("  Usage: python test_document_loader.py")
        return
    
    results = []
    
    # Run tests
    results.append(("Load single regulation", test_single_regulation()))
    results.append(("Load single paper", test_single_paper()))
    results.append(("Load small batch", test_load_small_batch()))
    results.append(("Load full dataset", test_full_load()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    total_pass = sum(1 for _, passed in results if passed)
    print(f"\nTotal: {total_pass}/{len(results)} tests passed")
    
    if total_pass == len(results):
        print("\nAll tests passed! Document loader is ready to use.")
      
    else:
        print("\nSome tests failed. Please review the errors above.")


if __name__ == "__main__":
    main()