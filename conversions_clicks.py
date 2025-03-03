import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from load_files import load_and_filter_by_rank, load_and_filter_by_terms, load_and_filter_initial_file, load_and_filter_subsequent_files, rename_columns

pd.set_option('future.no_silent_downcasting', True)



def find_conversions(row: pd.Series, prev_avg_orders: float = 1) -> Tuple[Dict[str, int], float]:
    """
    Calculates conversion metrics for a given row.
    """
    row = row.copy()
    row.loc[['Conversion_Share_1', 'Conversion_Share_2', 'Conversion_Share_3']] = (
        row.loc[['Conversion_Share_1', 'Conversion_Share_2', 'Conversion_Share_3']]
        .fillna(0)
        .infer_objects(copy=False)
    )
    
    # Check if all conversion shares are zero
    if (row['Conversion_Share_1'] == 0 and 
        row['Conversion_Share_2'] == 0 and 
        row['Conversion_Share_3'] == 0):
        return {
            'Total_Orders': 0,
            'Other_Orders': 0,
            'ASIN1_Orders': 0,
            'ASIN2_Orders': 0,
            'ASIN3_Orders': 0, 
            'SFR': row['SFR'] if 'SFR' in row else None
        }, 0

    # Calculate conversion shares and ratios
    conv_shares = [row['Conversion_Share_1'], 
                  row['Conversion_Share_2'], 
                  row['Conversion_Share_3']]
    min_share = min(share for share in conv_shares if share > 0)
    ratios = [share/min_share if share > 0 else 0 for share in conv_shares]
    
    # Calculate minimum order threshold based on previous month's data
    if row['SFR'] <= 5000:
        min_order_threshold = max(1, prev_avg_orders * 0.9)
    elif row['SFR'] <= 10000 and row['SFR'] > 5000:
        min_order_threshold = max(1, prev_avg_orders * 0.85)
    elif row['SFR'] > 10000 and row['SFR'] <= 40000:
        min_order_threshold = max(1, prev_avg_orders * 0.8)
    else:
        min_order_threshold = max(1, prev_avg_orders * 0.7)
    # Default values
    delta = 0.025
    start = 1
    
    # Determine starting value based on SFR and min_share
    # Using a lookup table approach for cleaner logic
    if row['SFR'] <= 200 and min_share >= 1:
        start = 3
    elif 200 < row['SFR'] <= 2000 and min_share >= 1:
        start = 2
    
    # Determine delta and start values based on SFR and min_share using a lookup table
    sfr_share_config = [
        # SFR range, min_share range, has_zero_share, delta, start
        ((0, 1700), (20, float('inf')), True, 0.005, 20),
        ((0, 1700), (8, 20), True, 0.005, 15),
        ((0, 1700), (2, 8), True, 0.005, 12),
        ((0, 1700), (1, 2), True, 0.005, 5),
        
        ((1700, 4000), (20, float('inf')), True, 0.01, 10),
        ((1700, 4000), (2, 20), True, 0.01, 6),
        ((1700, 4000), (1, 2), True, 0.01, 2),
        
        ((4000, 10000), (2, float('inf')), True, 0.01, 8),
        ((4000, 10000), (1, 2), True, 0.01, 2),
        
        ((10000, 40000), (2.5, float('inf')), True, 0.01, 2),
        
        ((40000, 100000), (9, float('inf')), True, 0.01, 2)
    ]
    
    # Special case for high min_share values
    if any(share == 0 for share in conv_shares) and min_share >= 50:
        if row['SFR'] <= 10000:
            delta, start = 0.005, 30
        elif 10000 < row['SFR'] <= 30000:
            delta, start = 0.005, 20
        elif 30000 < row['SFR'] <= 70000:
            delta, start = 0.005, 10
    
    # Apply configuration from lookup table if applicable
    has_zero_share = any(share == 0 for share in conv_shares)
    for sfr_range, share_range, requires_zero, d, s in sfr_share_config:
        if (sfr_range[0] < row['SFR'] <= sfr_range[1] and
            share_range[0] <= min_share < share_range[1] and
            (requires_zero == has_zero_share)):
            delta, start = d, s
            break
    
    # Determine a and b using lookup table
    ab_lookup = [
        # SFR range, a, b
        ((0, 100), 1800, 80),
        ((100, 300), 1500, 80),
        ((300, 500), 1250, 80),
        ((500, 1000), 1050, 65),
        ((1000, 2000), 900, 60),
        ((2000, 2700), 680, 50),
        ((2700, 3500), 500, 50),
        ((3500, 5000), 430, 45),
        ((5000, 15000), 375, 40),
        ((15000, 35000), 270, 35),
        ((35000, float('inf')), 220, 30)
    ]
    
    a, b = 220, 30  # Default values
    for sfr_range, a_val, b_val in ab_lookup:
        if sfr_range[0] < row['SFR'] <= sfr_range[1]:
            a, b = a_val, b_val
            break
            
    # Find valid order combinations
    while delta <= 3:
        for i in range(start, b):
            predicted = [ratio * i for ratio in ratios]
            
            if all(abs(pred - round(pred)) < delta for pred in predicted if pred > 0):
                orders = [round(pred) for pred in predicted]
                sum_ratio = sum(conv_shares)
                other_orders = round((100-sum_ratio)*sum(orders)/sum_ratio) if sum_ratio > 0 else 0
                total_orders = sum(orders) + other_orders
                
                if total_orders > a and delta < 1.5:
                    break
                    
                if (total_orders % 1000 != 0 and 
                    total_orders >= min_order_threshold):
                    return {
                        'Total_Orders': total_orders,
                        'Other_Orders': other_orders,
                        'ASIN1_Orders': orders[0],
                        'ASIN2_Orders': orders[1],
                        'ASIN3_Orders': orders[2],
                        'SFR': row['SFR'] if 'SFR' in row else None
                    }, delta
        delta += 0.02
        
    # If no valid combination found
    return {
        'Total_Orders': 0,
        'Other_Orders': 0,
        'ASIN1_Orders': 0,
        'ASIN2_Orders': 0,
        'ASIN3_Orders': 0,
        'SFR': row['SFR'] if 'SFR' in row else None
    }, delta


def find_clicks(row: pd.Series, conv_row: pd.Series, prev_avg_clicks: float = 1) -> Tuple[Dict[str, int], float]:
    """
    Calculates click metrics for a given row based on conversion data.
    """
    click_shares = [
        row['Click_Share_1'], 
        row['Click_Share_2'], 
        row['Click_Share_3']
    ]
    
    # Handle case when all click shares are zero
    if all(share == 0 for share in click_shares):
        return {
            'Total_Clicks': 1,
            'Other_Clicks': 1,
            'ASIN1_Clicks': 0,
            'ASIN2_Clicks': 0,
            'ASIN3_Clicks': 0
        }, 0
        
    min_click_share = min(share for share in click_shares if share > 0)
    click_ratios = [share/min_click_share if share > 0 else 0 for share in click_shares]
    
    # Calculate minimum click threshold based on previous month's data
    min_click_threshold = max(1, prev_avg_clicks * 0.85)
    
    # Define a lookup table for parameters based on SFR
    click_params = [
        # SFR range, a (max iterations), b (clicks to orders ratio), delta
        ((0, 100), 500, 4, 0.015),
        ((100, 500), 400, 3.6, 0.15),
        ((500, 1000), 300, 3.4, 0.015),
        ((1000, 5000), 200, 3.1, 0.02),
        ((5000, 20000), 140, 2.8, 0.025),
        ((20000, 50000), 100, 2, 0.025),
        ((50000, 100000), 80, 1.6, 0.025),
        ((100000, float('inf')), 60, 1, 0.025)
    ]
    
    # Get parameters from lookup table
    a, b, delta = 60, 1, 0.025  # Default values
    for sfr_range, a_val, b_val, d_val in click_params:
        if sfr_range[0] < row['SFR'] <= sfr_range[1]:
            a, b, delta = a_val, b_val, d_val
            break
    
    # Find valid click combinations
    while delta <= 3:
        for i in range(1, a):
            predicted_clicks = [ratio * i for ratio in click_ratios]
            
            if all(abs(click - round(click)) < delta for click in predicted_clicks if click > 0):
                clicks = [round(click) for click in predicted_clicks]
                sum_ratio = sum(click_shares)
                other_clicks = round((100-sum_ratio)*sum(clicks)/sum_ratio) if sum_ratio > 0 else 0
                total_clicks = sum(clicks) + other_clicks
                
                if (total_clicks % 1000 != 0 and
                    total_clicks >= min_click_threshold and
                    total_clicks >= conv_row['Total_Orders'] * b and
                    other_clicks >= conv_row['Other_Orders'] and
                    all(c >= o for c, o in zip(clicks, [
                        conv_row['ASIN1_Orders'],
                        conv_row['ASIN2_Orders'],
                        conv_row['ASIN3_Orders']
                    ]))):
                    
                    return {
                        'Total_Clicks': total_clicks,
                        'Other_Clicks': other_clicks,
                        'ASIN1_Clicks': clicks[0],
                        'ASIN2_Clicks': clicks[1],
                        'ASIN3_Clicks': clicks[2]
                    }, delta
        delta += 0.02
        if delta > 2.8:
            a += 50
            
    # If no valid combination found
    return {
        'Total_Clicks': 1,
        'Other_Clicks': 1,
        'ASIN1_Clicks': 0,
        'ASIN2_Clicks': 0,
        'ASIN3_Clicks': 0
    }, delta


def load_previous_data(
    file_path: str,
    columns_to_import: List[str]
    ) -> pd.DataFrame:
    """
    Loads data from previous month
    """
    try:
        data = pd.read_excel(
            file_path,
            usecols=columns_to_import
        )
    except Exception as e:
        print(f"Failed to load data from {file_path}: {e}")
        data = pd.DataFrame(columns=columns_to_import)
    return data

    
def process_search_term(
    search_term: str,
    all_dfs: List[pd.DataFrame],
    dates: List[datetime],
    previous_month_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Processes a single search term across all DataFrames and returns results.
    """
    # Get previous month's data for this search term
    prev_data_row = previous_month_data[previous_month_data['Search_Term'] == search_term]
    prev_avg_orders = prev_data_row['Average_Orders'].iloc[0] if not prev_data_row.empty else 1
    prev_avg_clicks = prev_data_row['Average_Clicks'].iloc[0] if not prev_data_row.empty else 1
    
    conversion_results = []
    click_results = []
    
    for df, date in zip(all_dfs, dates):
        row = df[df['Search_Term'] == search_term]
        if row.empty:
            conv_result = {
                'Total_Orders': 0,
                'Other_Orders': 0,
                'ASIN1_Orders': 0,
                'ASIN2_Orders': 0,
                'ASIN3_Orders': 0,
                'SFR': 0
            }
            click_result = {
                'Total_Clicks': 0,
                'Other_Clicks': 0,
                'ASIN1_Clicks': 0,
                'ASIN2_Clicks': 0,
                'ASIN3_Clicks': 0
            }
        else:
            conv_result, _ = find_conversions(row.iloc[0], prev_avg_orders)
            click_result, _ = find_clicks(row.iloc[0], pd.Series(conv_result), prev_avg_clicks)
        
        conversion_results.append(conv_result)
        click_results.append(click_result)
    
    conv_df = pd.DataFrame(conversion_results, index=dates)
    click_df = pd.DataFrame(click_results, index=dates)
    conv_df['Search_Term'] = search_term
    result_df = pd.concat([conv_df, click_df], axis=1)
    cols = ['Search_Term', 'SFR'] + [col for col in result_df.columns if col not in ['Search_Term', 'SFR']]
    result_df = result_df[cols]
    
    return result_df


def analyze_search_terms(
    initial_file: str,
    subsequent_files: List[str],
    columns_to_import: List[str],
    column_mapping: Dict[str, str],
    start_date: str,
    previous_month_file: Optional[str] = None,
    search_term_filter: Optional[str] = None,
    rank_range: Optional[Tuple[int, int]] = None,
    search_list: Optional[List[str]] = None
) -> List[pd.DataFrame]:
    """
    Main function to analyze search terms across multiple files.
    """
    # Load previous month data
    previous_month_data = load_previous_data(
        previous_month_file,
        ["Search_Term", "Average_Orders", "Average_Clicks"]
    ) if previous_month_file else pd.DataFrame(columns=["Search_Term", "Average_Orders", "Average_Clicks"])
    
    # Initialize initial DataFrame based on filter conditions
    if search_list is not None:
        # Priority 1: Filter by exact matches if search_list is provided
        initial_df = load_and_filter_by_terms(
            initial_file,
            columns_to_import,
            search_list
        )
    elif rank_range is not None:
        # Priority 2: Filter by rank range if provided
        initial_df = load_and_filter_by_rank(
            initial_file,
            columns_to_import,
            rank_range
        )
    elif search_term_filter is not None:
        # Priority 3: Filter by partial match if provided
        initial_df = load_and_filter_initial_file(
            initial_file,
            columns_to_import,
            search_term_filter
        )
    else:
        # If no filters provided, raise an error
        raise ValueError("At least one filter condition (search_list, rank_range, or search_term_filter) must be provided")
    
    # Get unique search terms
    filtered_terms = initial_df['Search Term'].unique()
    
    # Load and process subsequent files
    all_dfs = [initial_df] + load_and_filter_subsequent_files(
        subsequent_files,
        columns_to_import,
        filtered_terms
    )
    
    # Rename columns in all DataFrames
    all_dfs = [rename_columns(df, column_mapping) for df in all_dfs]
    
    # Generate dates
    dates = pd.date_range(start=start_date, freq='D', periods=len(all_dfs))
    
    # Process each search term
    result_dfs = []
    for search_term in filtered_terms:
        result_df = process_search_term(search_term, all_dfs, dates, previous_month_data)
        result_dfs.append(result_df)
    
    return result_dfs

