import pandas as pd
from utils.config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split a text into overlapping word chunks.

    Args:
        text (str): The input text.
        chunk_size (int): Number of words per chunk.
        overlap (int): Number of words overlapped between consecutive chunks.

    Returns:
        list[str]: List of text chunks.
    """
    if pd.isna(text):
        return []

    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(' '.join(chunk))
        i += chunk_size - overlap
    return chunks


def chunk_dataframe(df: pd.DataFrame, text_col: str = "description", id_col: str = "scopus_id") -> pd.DataFrame:
    """
    Apply chunking to a DataFrame column and return a new DataFrame
    with (scopus_id, chunk_id, chunk_text).

    Args:
        df (pd.DataFrame): Input DataFrame with text and ID columns.
        text_col (str): Column name containing the text to chunk.
        id_col (str): Column name containing the document ID.

    Returns:
        pd.DataFrame: Chunked DataFrame with scopus_id, chunk_id, chunk_text.
    """
    chunks_series = df[text_col].apply(chunk_text)
    chunk_records = []

    for doc_id, doc_chunks in zip(df[id_col], chunks_series):
        for i, chunk in enumerate(doc_chunks):
            chunk_records.append({
                id_col: doc_id,
                "chunk_id": i,
                "chunk_text": chunk
            })

    return pd.DataFrame(chunk_records)
