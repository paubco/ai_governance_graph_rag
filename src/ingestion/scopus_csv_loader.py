# -*- coding: utf-8 -*-
"""
SCOPUS CSV LOADER
Loads and cleans Scopus export CSV with full metadata
INPUT: Raw Scopus CSV export (with UTF-8 BOM)
OUTPUT: Clean structured CSV with metadata
"""
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional
import sys


class ScopusCSVLoader:
    """
    Loads Scopus export CSV and structures metadata for downstream processing.
    
    Key Features:
    - Handles UTF-8 BOM encoding
    - Parses complex fields (authors, affiliations, keywords, funding)
    - Cleans and validates data
    - Generates clean intermediate format
    """
    
    def __init__(self, csv_path: str, output_dir: str = "data/interim"):
        """
        Initialize loader.
        
        Args:
            csv_path: Path to Scopus CSV export
            output_dir: Where to save cleaned data
        """
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = self._setup_logging()
        
        # Will hold the dataframe
        self.df = None
        
    def _setup_logging(self) -> logging.Logger:
        """Configure logging for this module"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def load(self) -> pd.DataFrame:
        """
        Load and parse Scopus CSV.
        
        Returns:
            DataFrame with cleaned metadata
        """
        self.logger.info(f"Loading Scopus CSV from: {self.csv_path}")
        
        # Handle UTF-8 BOM encoding (Scopus exports have this)
        try:
            self.df = pd.read_csv(
                self.csv_path,
                encoding='utf-8-sig',  # Handles BOM
                on_bad_lines='warn'     # Don't crash on malformed lines
            )
            self.logger.info(f"Successfully loaded {len(self.df)} papers")
        except Exception as e:
            self.logger.error(f"Failed to load CSV: {e}")
            raise
        
        return self.df
    
    def clean_and_structure(self) -> pd.DataFrame:
        """
        Clean and structure the loaded data.
        
        Extracts key fields:
        - Basic: EID, Title, Year, DOI
        - Authors: Names, IDs, Affiliations
        - Content: Abstract, Keywords
        - Metadata: Citation count, Funding, Language
        
        Returns:
            Cleaned DataFrame
        """
        if self.df is None:
            raise ValueError("Must call load() first")
        
        self.logger.info("Cleaning and structuring data...")
        
        # Select and rename key columns
        structured_df = pd.DataFrame()
        
        # Basic identifiers
        structured_df['scopus_id'] = self.df['EID']
        structured_df['doi'] = self.df['DOI'].fillna('')
        structured_df['title'] = self.df['Title']
        structured_df['year'] = self.df['Year']
        
        # Authors (simple list, semicolon-separated)
        structured_df['authors'] = self.df['Authors'].fillna('')
        structured_df['author_ids'] = self.df['Author(s) ID'].fillna('')
        
        # Affiliations (complex field - keep as-is for now)
        structured_df['affiliations'] = self.df['Affiliations'].fillna('')
        
        # Content
        structured_df['abstract'] = self.df['Abstract'].fillna('')
        
        # Keywords (both types)
        structured_df['author_keywords'] = self.df['Author Keywords'].fillna('')
        structured_df['index_keywords'] = self.df['Index Keywords'].fillna('')
        
        # Publication details
        structured_df['source'] = self.df['Source title'].fillna('')
        structured_df['volume'] = self.df['Volume'].fillna('')
        structured_df['issue'] = self.df['Issue'].fillna('')
        
        # Metrics
        structured_df['cited_by'] = pd.to_numeric(
            self.df['Cited by'], errors='coerce'
        ).fillna(0).astype(int)
        
        # Funding (may be important for entity extraction)
        structured_df['funding'] = self.df['Funding Details'].fillna('')
        
        # Language
        structured_df['language'] = self.df['Language of Original Document'].fillna('English')
        
        # Document type
        structured_df['doc_type'] = self.df['Document Type'].fillna('Article')
        
        # Open access status
        structured_df['open_access'] = self.df['Open Access'].fillna('')
        
        self.logger.info("Data cleaning complete")
        self.logger.info(f"Columns extracted: {list(structured_df.columns)}")
        
        return structured_df
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Run validation checks on the cleaned data.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            Dictionary with validation statistics
        """
        self.logger.info("Running data validation...")
        
        stats = {
            'total_papers': len(df),
            'papers_with_abstract': (df['abstract'] != '').sum(),
            'papers_without_abstract': (df['abstract'] == '').sum(),
            'papers_with_doi': (df['doi'] != '').sum(),
            'papers_with_keywords': (
                (df['author_keywords'] != '') | (df['index_keywords'] != '')
            ).sum(),
            'papers_with_funding': (df['funding'] != '').sum(),
            'papers_with_affiliations': (df['affiliations'] != '').sum(),
            'unique_years': df['year'].nunique(),
            'year_range': (df['year'].min(), df['year'].max()),
            'languages': df['language'].value_counts().to_dict(),
            'total_citations': df['cited_by'].sum(),
            'avg_citations': df['cited_by'].mean()
        }
        
        # Log key statistics
        self.logger.info("=" * 60)
        self.logger.info("VALIDATION STATISTICS")
        self.logger.info("=" * 60)
        self.logger.info(f"Total papers: {stats['total_papers']}")
        self.logger.info(f"Papers with abstracts: {stats['papers_with_abstract']} "
                        f"({stats['papers_with_abstract']/stats['total_papers']*100:.1f}%)")
        self.logger.info(f"Papers WITHOUT abstracts: {stats['papers_without_abstract']}")
        self.logger.info(f"Papers with DOI: {stats['papers_with_doi']} "
                        f"({stats['papers_with_doi']/stats['total_papers']*100:.1f}%)")
        self.logger.info(f"Papers with keywords: {stats['papers_with_keywords']}")
        self.logger.info(f"Papers with funding info: {stats['papers_with_funding']}")
        self.logger.info(f"Year range: {stats['year_range'][0]} - {stats['year_range'][1]}")
        self.logger.info(f"Total citations: {stats['total_citations']}")
        self.logger.info(f"Average citations: {stats['avg_citations']:.2f}")
        self.logger.info("=" * 60)
        
        # Warnings for potential issues
        if stats['papers_without_abstract'] > 0:
            self.logger.warning(
                f"{stats['papers_without_abstract']} papers have no abstract! "
                "These will need PDF text extraction."
            )
        
        return stats
    
    def save(self, df: pd.DataFrame, filename: str = "scopus_metadata.csv") -> Path:
        """
        Save cleaned data to CSV with proper quoting.
        
        Args:
            df: Cleaned DataFrame
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        
        self.logger.info(f"Saving cleaned data to: {output_path}")
        
        # Use QUOTE_ALL to handle commas in text properly
        df.to_csv(
            output_path, 
            index=False, 
            encoding='utf-8',
            quoting=1  # csv.QUOTE_ALL - quotes all fields
        )
        
        self.logger.info(f"Successfully saved {len(df)} papers")
        
        return output_path
    
    def run_full_pipeline(self) -> tuple[pd.DataFrame, Dict, Path]:
        """
        Run the complete loading pipeline.
        
        Returns:
            Tuple of (cleaned_df, validation_stats, output_path)
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING SCOPUS CSV LOADING PIPELINE")
        self.logger.info("=" * 60)
        
        # Step 1: Load
        self.load()
        
        # Step 2: Clean and structure
        cleaned_df = self.clean_and_structure()
        
        # Step 3: Validate
        stats = self.validate_data(cleaned_df)
        
        # Step 4: Save
        output_path = self.save(cleaned_df)
        
        self.logger.info("=" * 60)
        self.logger.info("PIPELINE COMPLETE")
        self.logger.info("=" * 60)
        
        return cleaned_df, stats, output_path


def main():
    """
    Example usage and testing
    """
    import sys
    from pathlib import Path
    
    # Allow year to be specified as command-line argument
    year = sys.argv[1] if len(sys.argv) > 1 else "2023"
    
    # Use standardized naming convention
    # Expected filename format: scopus_export_{YEAR}_raw.csv
    raw_data_dir = Path("data/raw/academic") / f"scopus_{year}"
    csv_filename = f"scopus_export_{year}_raw.csv"
    csv_path = raw_data_dir / csv_filename
    
    # Check if file exists
    if not csv_path.exists():
        print(f"ERROR: Expected file not found: {csv_path}")
        print(f"\nPlease ensure you have:")
        print(f"1. Downloaded Scopus export for {year}")
        print(f"2. Renamed it to: {csv_filename}")
        print(f"3. Placed it in: {raw_data_dir}/")
        print(f"\nOriginal Scopus exports have random UUIDs in filenames.")
        print(f"Rename them immediately after download for reproducibility.")
        sys.exit(1)
    
    output_filename = f"scopus_metadata_{year}.csv"
    
    loader = ScopusCSVLoader(
        csv_path=str(csv_path),
        output_dir="data/interim/academic"
    )
    
    # Run full pipeline
    df, stats, output_path = loader.run_full_pipeline()
    
    # Save with year-specific filename
    output_path = loader.save(df, filename=output_filename)
    
    print(f"\n{'='*60}")
    print(f"SUCCESS: Dataset processed")
    print(f"{'='*60}")
    print(f"Input:  {csv_path}")
    print(f"Output: {output_path}")
    print(f"Year:   {year}")
    print(f"Papers: {len(df)}")
    print(f"{'='*60}")
    
    return df, stats

if __name__ == "__main__":
    main()