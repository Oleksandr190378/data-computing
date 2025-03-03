from typing import List
import pandas as pd

def adjust_clicks_and_orders(results: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Adjusts anomalous clicks and orders in each DataFrame from the results list.
    """
    adjusted_results = []
    for df in results:
        # Create a copy of the DataFrame to avoid modifying the original
        adjusted_df = df.copy()
        # First adjust clicks
        adjusted_df = adjust_clicks(adjusted_df)
        # Then adjust orders based on adjusted clicks
        adjusted_df = adjust_orders(adjusted_df)
        adjusted_results.append(adjusted_df)
    return adjusted_results


def get_valid_median(series: pd.Series, index, index_min) -> float:
    """
    Calculate median excluding zeros and NaN values.
    """
    valid_values = series[(series > index_min) & (series <= index)].dropna()
    return valid_values.mean() if not valid_values.empty else 0


def get_valid_median_clicks(series: pd.Series, index) -> float:
    """
    Calculate median excluding zeros and NaN values.
    """
    valid_values = series[series > index].dropna()
    return valid_values.mean() if not valid_values.empty else 0


def get_click_thresholds(sfr: float):
    """
    Return index threshold for clicks based on SFR value.
    """
    sfr = round(sfr)
    
    if sfr <= 100:
        return 1000
    elif sfr <= 300:
        return 900
    elif sfr <= 500:
        return 800 
    elif sfr <= 1000:
        return 750
    elif sfr <= 2000:
        return 700
    elif sfr <= 2700:
        return 700  
    elif sfr <= 3500:
        return 600 
    elif sfr <= 10000:
        return 500
    elif sfr <= 20000:
        return 300
    elif sfr <= 50000:
        return 200  
    elif sfr <= 100000:
        return 80
    else:
        return 10


def get_order_thresholds(sfr: float):
    """
    Return index and index_min thresholds for orders based on SFR value.
    """
    sfr = round(sfr)
    
    if sfr <= 100:
        return 1700, 300
    elif sfr <= 330:
        return 1500, 290
    elif sfr <= 500:
        return 1350, 270
    elif sfr <= 1000:
        return 1200, 240
    elif sfr <= 2000:
        return 950, 180
    elif sfr <= 2700:
        return 700, 140 
    elif sfr <= 3500:
        return 550, 130
    elif sfr <= 5000:
        return 430, 110
    elif sfr <= 10000:
        return 370, 80
    elif sfr <= 20000:
        return 310, 45
    elif sfr <= 35000:
        return 230, 20
    else:
        return 150, 0


def get_click_bounds(sfr: float, median_clicks: float):
    """
    Calculate bounds for click adjustments based on SFR.
    """
    sfr = round(sfr)
    
    if sfr <= 4100:
        lower_bound = 0.9 * median_clicks
        upper_bound = 1.7 * median_clicks
        target_lower = 0.9 * median_clicks
        target_upper = 1.7 * median_clicks
    elif sfr <= 10000:
        lower_bound = 0.85 * median_clicks
        upper_bound = 1.6 * median_clicks
        target_lower = 0.85 * median_clicks
        target_upper = 1.6 * median_clicks
    elif sfr <= 20000:
        lower_bound = 0.8 * median_clicks
        upper_bound = 1.5 * median_clicks
        target_lower = 0.8 * median_clicks
        target_upper = 1.5 * median_clicks
    else:
        lower_bound = 0.75 * median_clicks
        upper_bound = 1.5 * median_clicks
        target_lower = 0.75 * median_clicks
        target_upper = 1.5 * median_clicks
        
    return lower_bound, upper_bound, target_lower, target_upper


def get_order_bounds(sfr: float, median_orders: float):
    """
    Calculate bounds for order adjustments based on SFR.
    """
    sfr = round(sfr)
    
    if sfr <= 4100:
        lower_bound = 0.8 * median_orders
        upper_bound = 1.4 * median_orders
        target_lower = 0.8 * median_orders
        target_upper = 1.4 * median_orders
    elif sfr <= 10000:
        lower_bound = 0.75 * median_orders
        upper_bound = 1.4 * median_orders
        target_lower = 0.75 * median_orders
        target_upper = 1.4 * median_orders
    elif sfr <= 20000:
        lower_bound = 0.7 * median_orders
        upper_bound = 1.4 * median_orders
        target_lower = 0.7 * median_orders
        target_upper = 1.4 * median_orders
    else:
        lower_bound = 0.6 * median_orders
        upper_bound = 1.4 * median_orders
        target_lower = 0.6 * median_orders
        target_upper = 1.4 * median_orders
        
    return lower_bound, upper_bound, target_lower, target_upper

    
def adjust_clicks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust anomalous click values.
    """
    sfr = round(df['SFR'].mean())
    index = get_click_thresholds(sfr)
    
    median_clicks = get_valid_median_clicks(df['Total_Clicks'], index)
    if median_clicks == 0:
        return df
        
    lower_bound, upper_bound, target_lower, target_upper = get_click_bounds(sfr, median_clicks)
    
    for idx in df.index:
        clicks = df.at[idx, 'Total_Clicks']
        if clicks == 0:
            continue
        if clicks == 1:
            df.at[idx, 'Total_Clicks'] = round(median_clicks)
        if clicks < lower_bound:
            # Try multiplying
            for multiplier in range(2, 51):
                new_clicks = clicks * multiplier
                if target_lower <= new_clicks <= target_upper:
                    ratio = multiplier
                    df.at[idx, 'Total_Clicks'] = new_clicks
                    df.at[idx, 'Other_Clicks'] = df.at[idx, 'Other_Clicks'] * ratio
                    df.at[idx, 'ASIN1_Clicks'] = df.at[idx, 'ASIN1_Clicks'] * ratio
                    df.at[idx, 'ASIN2_Clicks'] = df.at[idx, 'ASIN2_Clicks'] * ratio
                    df.at[idx, 'ASIN3_Clicks'] = df.at[idx, 'ASIN3_Clicks'] * ratio
                    break    
        elif clicks > upper_bound:
            # Try dividing
            for divisor in range(2, 51):
                new_clicks = clicks / divisor
                if target_lower <= new_clicks <= target_upper:
                    ratio = 1/divisor
                    df.at[idx, 'Total_Clicks'] = round(new_clicks)
                    df.at[idx, 'Other_Clicks'] = round(df.at[idx, 'Other_Clicks'] * ratio)
                    df.at[idx, 'ASIN1_Clicks'] = round(df.at[idx, 'ASIN1_Clicks'] * ratio)
                    df.at[idx, 'ASIN2_Clicks'] = round(df.at[idx, 'ASIN2_Clicks'] * ratio)
                    df.at[idx, 'ASIN3_Clicks'] = round(df.at[idx, 'ASIN3_Clicks'] * ratio)
                    break
    
    return df


def adjust_orders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust anomalous order values while maintaining relationship with clicks.
    """
    sfr = round(df['SFR'].mean())
    index, index_min = get_order_thresholds(sfr)
    
    median_orders = get_valid_median(df['Total_Orders'], index, index_min)
    if median_orders == 0:
        return df
        
    lower_bound, upper_bound, target_lower, target_upper = get_order_bounds(sfr, median_orders)
    
    for idx in df.index:
        orders = df.at[idx, 'Total_Orders']
        clicks = df.at[idx, 'Total_Clicks']
        if orders == 0 or clicks == 0:
            continue
        if orders < lower_bound:
            # Try multiplying
            for multiplier in range(2, 51):
                new_orders = orders * multiplier
                if (target_lower <= new_orders <= target_upper and 
                    new_orders <= clicks):
                    ratio = multiplier
                    df.at[idx, 'Total_Orders'] = new_orders
                    df.at[idx, 'Other_Orders'] = df.at[idx, 'Other_Orders'] * ratio
                    df.at[idx, 'ASIN1_Orders'] = df.at[idx, 'ASIN1_Orders'] * ratio
                    df.at[idx, 'ASIN2_Orders'] = df.at[idx, 'ASIN2_Orders'] * ratio
                    df.at[idx, 'ASIN3_Orders'] = df.at[idx, 'ASIN3_Orders'] * ratio
                    break    
        elif orders > upper_bound:
            # Try dividing
            for divisor in range(2, 51):
                new_orders = orders / divisor
                if (target_lower <= new_orders <= target_upper and 
                    new_orders <= clicks):
                    ratio = 1/divisor
                    df.at[idx, 'Total_Orders'] = round(new_orders)
                    df.at[idx, 'Other_Orders'] = round(df.at[idx, 'Other_Orders'] * ratio)
                    df.at[idx, 'ASIN1_Orders'] = round(df.at[idx, 'ASIN1_Orders'] * ratio)
                    df.at[idx, 'ASIN2_Orders'] = round(df.at[idx, 'ASIN2_Orders'] * ratio)
                    df.at[idx, 'ASIN3_Orders'] = round(df.at[idx, 'ASIN3_Orders'] * ratio)
                    break
    
    return df

