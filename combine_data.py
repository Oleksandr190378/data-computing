import pandas as pd
#from datetime import datetime, date

def create_combined_dataframe(list_of_dataframes: list) -> pd.DataFrame:
    """
    Combines a list of DataFrames into a single DataFrame, aggregating search term data 
    across different dates.

    Each input DataFrame is expected to have a 'Search_Term' column and be indexed by date.
    For each DataFrame, the function extracts the 'SFR', 'Total_Orders', and 'Total_Clicks'
    values and adds them as columns to the resulting DataFrame with date-specific column names.

    Parameters
    ----------
    list_of_dataframes : list
        A list of pandas DataFrames, each containing search term data indexed by date.
    
    Returns
    -------
    pd.DataFrame
        A combined DataFrame with aggregated search term data. Columns include 'Search_Term',
        as well as 'SFR', 'Orders', and 'Clicks' for each date. NaN values in 'SFR' columns
        are filled with 0 and converted to integers.
    """

    combined_data = []
    for df in list_of_dataframes:
        row_data = {'Search_Term': df['Search_Term'].iloc[0]}  
        for date in df.index:
            row_data[f'SFR {date.date()}'] = df.loc[date, 'SFR']
            row_data[f'Orders {date.date()}'] = df.loc[date, 'Total_Orders']
            row_data[f'Clicks {date.date()}'] = df.loc[date, 'Total_Clicks']

        combined_data.append(row_data)
    combined_df = pd.DataFrame(combined_data)    
    for column in combined_df.columns:
        if 'SFR' in column:
            combined_df[column] = combined_df[column].fillna(0).astype(int)
            
    return combined_df

