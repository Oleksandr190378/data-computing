import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional


#pd.set_option('future.no_silent_downcasting', True)


def load_and_filter_by_rank(
    file_path: str,
    columns_to_import: List[str],
    rank_range: Tuple[int, int],
    chunk_size: int = 10000
) -> pd.DataFrame:
    """
    Loads and filters the initial file for search terms within specified SFR (Search Frequency Rank) range.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    columns_to_import : List[str]
        List of column names to import
    rank_range : Tuple[int, int]
        Tuple of (min_rank, max_rank) to filter by
    chunk_size : int, optional
        Size of chunks for reading large files, default is 10000
        
    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame containing only rows where SFR is within the specified range
    """
    def filter_chunks(chunk: pd.DataFrame) -> pd.DataFrame:
        # Convert SFR to numeric, coerce errors to NaN
        chunk['Search Frequency Rank'] = pd.to_numeric(chunk['Search Frequency Rank'], errors='coerce')
        # Filter by rank range and remove any NaN values
        return chunk[
            (chunk['Search Frequency Rank'] >= rank_range[0]) & 
            (chunk['Search Frequency Rank'] <= rank_range[1])
        ].dropna(subset=['Search Frequency Rank'])
    
    chunks = pd.read_csv(
        file_path,
        skiprows=1,
        usecols=columns_to_import,
        chunksize=chunk_size
    )
    
    filtered_df = pd.concat([filter_chunks(chunk) for chunk in chunks])
    
    # Sort by rank to ensure ordered results
    return filtered_df.sort_values('Search Frequency Rank')


def load_and_filter_by_terms(
    file_path: str,
    columns_to_import: List[str],
    search_terms: List[str],
    chunk_size: int = 10000
) -> pd.DataFrame:
    """
    Loads and processes file, ensuring one row per search term.
    If a search term isn't found, creates a default row with zeros for most columns
    and 1 for 'Top Clicked Product #X: Click Share' columns.
    
    Args:
        file_path (str): Path to the CSV file
        columns_to_import (List[str]): List of column names to import
        search_terms (List[str]): List of search terms to match exactly
        chunk_size (int): Size of chunks to process at once
        
    Returns:
        pd.DataFrame: DataFrame with one row per search term
    """
    # Read the entire file in chunks and combine
    chunks = pd.read_csv(
        file_path,
        skiprows=1,
        usecols=columns_to_import,
        chunksize=chunk_size
    )
    full_df = pd.concat(chunks)
    
    # Columns that should have default value 1
    top_click_columns = [
        'Top Clicked Product #1: Click Share',
        'Top Clicked Product #2: Click Share',
        'Top Clicked Product #3: Click Share'
    ]
    
    # Initialize empty list to store results
    result_rows = []
    
    # Process each search term
    for term in search_terms:
        # Find exact match
        matched_row = full_df[full_df['Search Term'] == term]
        
        if len(matched_row) > 0:
            # If match found, add the first matching row
            result_rows.append(matched_row.iloc[0])
        else:
            # Create default values for all columns
            default_values = {}
            default_values['Search Term'] = term
            
            # Assign default values for remaining columns
            for col in columns_to_import:
                if col == 'Search Term':
                    continue
                elif col in top_click_columns:
                    default_values[col] = 1  # Set 1 for top clicked product columns
                else:
                    default_values[col] = 0  # Set 0 for all other columns
            
            default_row = pd.Series(default_values)
            result_rows.append(default_row)
    
    # Combine all rows into final DataFrame
    result_df = pd.DataFrame(result_rows)
    
    # Ensure columns are in the same order as input
    result_df = result_df[columns_to_import]
    
    return result_df


def load_and_filter_initial_file(
    file_path: str,
    columns_to_import: List[str],
    search_term: str,
    chunk_size: int = 10000
) -> pd.DataFrame:
    """
    Loads and filters the initial file for specific search terms.
    """
    def filter_chunks(chunk: pd.DataFrame) -> pd.DataFrame:
        return chunk[chunk['Search Term'].str.contains(search_term, na=False)]
    
    chunks = pd.read_csv(
        file_path,
        skiprows=1,
        usecols=columns_to_import,
        chunksize=chunk_size
    )
    return pd.concat([filter_chunks(chunk) for chunk in chunks])


def load_and_filter_subsequent_files(
    file_paths: List[str],
    columns_to_import: List[str],
    filtered_terms: List[str],
    chunk_size: int = 10000
) -> List[pd.DataFrame]:
    """
    Loads and filters subsequent files based on search terms from initial file.
    """
    def filter_chunks(chunk: pd.DataFrame) -> pd.DataFrame:
        return chunk[chunk['Search Term'].isin(filtered_terms)]
    
    filtered_dfs = []
    for file_path in file_paths:
        chunks = pd.read_csv(
            file_path,
            skiprows=1,
            usecols=columns_to_import,
            chunksize=chunk_size
        )
        filtered_df = pd.concat([filter_chunks(chunk) for chunk in chunks])
        filtered_dfs.append(filtered_df)
    
    return filtered_dfs


def rename_columns(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Renames DataFrame columns according to the provided mapping.
    """
    return df.rename(columns=column_mapping)

