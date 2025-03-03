import pandas as pd
import os
from typing import List,Tuple, Optional
from datetime import date, timedelta
import argparse


pd.set_option('future.no_silent_downcasting', True)

def get_list(date_list: str) -> List[str]:

    """
    Reads a CSV file with search terms for a given date and returns a list of these terms.

    Parameters
    ----------
    date_list : str
        A string of the form 'MM-DD' representing the date of the search terms.

    Returns
    -------
    List[str]
        A list of search terms for the given date or an empty list if an error occurs.
    """

    base_path = r'C:\Users\user\Downloads\US_Top_Search_Terms_Simple_Day_'
    try:    
        # Перевіримо правильність введеного рядка дати
        value = date_list.split('-')
        if len(value) != 2:
            raise ValueError("Incorrect date format. Please use 'MM-DD'.")
        
        start_date = date(2025, int(value[0]), int(value[1]))
        
        # Формуємо шлях до файлу
        path_file = f"{base_path}{start_date.strftime('%Y_%m_%d')}.csv"
        
        # Завантажимо дані з файлу
        df = pd.read_csv(
            path_file,
            skiprows=1,
            usecols=['Search Term',]
        )
        
        return df['Search Term'].to_list()
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return []
    except FileNotFoundError:
        print(f"FileNotFoundError: The file {path_file} does not exist.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    

def get_file_paths(period: str, value: int) -> Tuple[str, List[str], str]:
    """
    Generate file paths and start date based on the period and value.
    
    :param period: 'week' or 'month'
    :param value: week number or month number
    :return: initial file path, list of subsequent file paths, start date
    """
    base_path = r'C:\Users\user\Downloads\US_Top_Search_Terms_Simple_Day_'
    start_date = None
    initial_file = None
    subsequent_files = []

    if period == 'month':
        start_date = date(2025, value, 1)
        initial_file = f"{base_path}{start_date.strftime('%Y_%m_%d')}.csv"
        for day in range(2, 32):
            try:
                file_date = date(2025, value, day)
                subsequent_files.append(f"{base_path}{file_date.strftime('%Y_%m_%d')}.csv")
            except ValueError:
                break

    elif period == 'week':
        start_date = date(2025, 1, 1) + timedelta(weeks=value-2)
        while start_date.weekday() != 6:  # Find Sunday
            start_date += timedelta(days=1)
        initial_file = f"{base_path}{start_date.strftime('%Y_%m_%d')}.csv"
        for day in range(1, 7):
            file_date = start_date + timedelta(days=day)
            subsequent_files.append(f"{base_path}{file_date.strftime('%Y_%m_%d')}.csv")

    start_date_str = start_date.strftime('%Y-%m-%d')
    return initial_file, subsequent_files, start_date_str



def get_previous_file(period: str, value: int) -> Optional[str]:
    """
    Determine the path to the previous period's file based on the current period and value.
    
    Parameters:
    -----------
    period : str
        Either 'month' or 'week'
    value : int
        The month number (1-12) or week number
        
    Returns:
    --------
    Optional[str]
        Path to the previous period's file if it exists, otherwise None
    """
    if period == 'month':
        # For months, simply decrement the month value
        prev_value = value - 1
        
        # Handle January (need to go to previous year's December)
        if prev_value < 1:
            prev_value = 12
            # Note: We're not handling the year change in the filename here
            # If needed, you could add a year parameter and adjust accordingly
    
    elif period == 'week':
        # For weeks, determine the month for the given week number
        current_year = date.today().year

        # Calculate the approximate date of the middle of the week
        week_start_date = date(current_year, 1, 1) + timedelta(weeks=value-1)
        middle_of_week_date = week_start_date + timedelta(days=3)

        # Determine the month of the middle of the week
        current_month = middle_of_week_date.month

        # Get the previous month
        prev_value = current_month - 1
        if prev_value < 1:
            prev_value = 12

    else:
        raise ValueError(f"Invalid period: {period}. Must be 'month' or 'week'.")

    # Construct the potential file path
    prev_file_path = f"C:\\Users\\user\\conversions_and_clicks_month_{prev_value}_{date.today().year}.json"
    
    # Check if the file exists
    if os.path.exists(prev_file_path):
        return prev_file_path
    else:
        # If direct previous doesn't exist, try with different extensions
        alt_extensions = ['.xlsx', '.csv']
        for ext in alt_extensions:
            alt_path = f"C:\\Users\\user\\conversions_and_clicks_{period}_{prev_value}{ext}"
            if os.path.exists(alt_path):
                return alt_path
        
        # If we can't find a previous month/week file, try the default previous month file
        default_prev_file = r"C:\Users\user\compare_with_analytics_2.xlsx"
        if os.path.exists(default_prev_file):
            return default_prev_file
            
        # Nothing found
        return None



if __name__ == "__main__":
    '''parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--date_list', type=str, help='Date in MM-DD format')
    parser.add_argument('--period', type=str, choices=['week', 'month'], help='Period (week or month)')
    parser.add_argument('--value', type=int, help='Week number or month number')

    args = parser.parse_args()

    if args.date_list:
        print(len(get_list(args.date_list)))
    elif args.period and args.value:
        print(get_file_paths(args.period, args.value))'''

    print(get_previous_file('week', 6))

        