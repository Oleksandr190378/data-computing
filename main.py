import pandas as pd
import numpy as np
import argparse
import sys
from datetime import date
from conversions_clicks import analyze_search_terms
from adjusting import adjust_clicks_and_orders
from combine_data import create_combined_dataframe
from process_data import process_orders, calculate_monthly_metrics, calculate_week_metrics
from get_file_path import get_file_paths, get_list

pd.set_option('future.no_silent_downcasting', True)

# Constants
previous_month_file = r"C:\Users\user\compare_with_analytics_2.xlsx"
columns_to_import = [
    'Search Frequency Rank', 'Search Term', 'Top Clicked Product #1: ASIN',
    'Top Clicked Product #1: Product Title', 'Top Clicked Product #1: Click Share',
    'Top Clicked Product #1: Conversion Share', 'Top Clicked Product #2: ASIN',
    'Top Clicked Product #2: Product Title', 'Top Clicked Product #2: Click Share',
    'Top Clicked Product #2: Conversion Share', 'Top Clicked Product #3: ASIN',
    'Top Clicked Product #3: Product Title', 'Top Clicked Product #3: Click Share',
    'Top Clicked Product #3: Conversion Share'
]

column_mapping = {
    'Search Frequency Rank': 'SFR',
    'Search Term': 'Search_Term',
    'Top Clicked Product #1: ASIN': 'ASIN_1',
    'Top Clicked Product #1: Product Title': 'Title_1',
    'Top Clicked Product #1: Click Share': 'Click_Share_1',
    'Top Clicked Product #1: Conversion Share': 'Conversion_Share_1',
    'Top Clicked Product #2: ASIN': 'ASIN_2',
    'Top Clicked Product #2: Product Title': 'Title_2',
    'Top Clicked Product #2: Click Share': 'Click_Share_2',
    'Top Clicked Product #2: Conversion Share': 'Conversion_Share_2',
    'Top Clicked Product #3: ASIN': 'ASIN_3',
    'Top Clicked Product #3: Product Title': 'Title_3',
    'Top Clicked Product #3: Click Share': 'Click_Share_3',
    'Top Clicked Product #3: Conversion Share': 'Conversion_Share_3'
}

def process_data(initial_file, subsequent_files, start_date_str, period, value, search_list=None):
    """Process data using the pipeline of functions."""
    try:
        # Step 1: Analyze search terms
        results_1 = analyze_search_terms(
            initial_file,
            subsequent_files,
            columns_to_import,
            column_mapping,
            start_date_str,
            previous_month_file,
            search_list=search_list
        )
        print(f"Number of results after analyze_search_terms: {len(results_1)}")
        
        # Step 2: Adjust clicks and orders
        results_2 = adjust_clicks_and_orders(results_1)
        
        # Step 3: Create combined dataframe
        combined_df = create_combined_dataframe(results_2)
        print("Preview of combined dataframe:")
        print(combined_df.head())
        
        # Step 4: Process orders and calculate metrics based on period
        combined_df = process_orders(combined_df)
        
        # Choose metrics calculation based on period
        if period == 'week':
            combined_df = calculate_week_metrics(combined_df)
        elif period == 'month':
            combined_df = calculate_monthly_metrics(combined_df)
        
        print(f"Final dataframe shape: {combined_df.shape}")
        
        # Save results with dynamic filename based on period and value
        output_path = f"C:\\Users\\user\\conversions_and_clicks_{period}_{value}_{date.today().year}.json"
        combined_df.to_json(output_path, orient='records', lines=True)
        print(f"Results saved to {output_path}")
        
        return combined_df
    
    except FileNotFoundError as e:
        print(f"Error: Could not find file. {e}")
        print("Please check that all file paths are correct and try again.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing data: {e}")
        sys.exit(1)

def print_usage():
    """Print usage instructions."""
    print("\nUsage:")
    print("1. Get search terms and process data:")
    print("   python main.py --date_list 02-15 [--limit 5] --period week --value 5")
    print("   python main.py --date_list 02-15 [--limit 5] --period month --value 2")
    print("2. Process data for a week:   python main.py --period week --value 5")
    print("3. Process data for a month:  python main.py --period month --value 2")
    print("4. Only get search terms:     python main.py --date_list 02-15 [--limit 5]")
    print("5. Manual file specification: python main.py --initial_file path/to/file.csv --subsequent_files file1.csv file2.csv --start_date 2025-02-02 --period week --value 5")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process search term data with various options.")
    
    # File selection options
    file_group = parser.add_argument_group('File Selection Options')
    file_group.add_argument('--date_list', type=str, help='Get search terms for date in MM-DD format')
    file_group.add_argument('--limit', type=int, default=10, help='Limit the number of search terms (default: 10)')
    file_group.add_argument('--period', type=str, choices=['week', 'month'], 
                           help='Process data for a week or month')
    file_group.add_argument('--value', type=int, 
                           help='Week number or month number to process')
    
    # Manually specified files
    file_group.add_argument('--initial_file', type=str, 
                           help='Path to initial file')
    file_group.add_argument('--subsequent_files', type=str, nargs='+', 
                           help='Paths to subsequent files')
    file_group.add_argument('--start_date', type=str, 
                           help='Start date in YYYY-MM-DD format')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output', type=str, 
                             help='Output file path')
    
    args = parser.parse_args()
    
    # Load search terms from previous month for filtering (default)
    try:
        df_game = pd.read_excel(previous_month_file)
        default_search_list = df_game['Search_Term'].to_list()
    except Exception as e:
        print(f"Warning: Could not load previous month data: {e}")
        default_search_list = None
    
    # Initialize search_list variable
    search_list = default_search_list
    
    # Process date_list argument if provided
    if args.date_list:
        try:
            # Get search terms for a specific date
            full_search_terms = get_list(args.date_list)
            if not full_search_terms:
                print(f"Error: Could not retrieve search terms for date {args.date_list}")
                print_usage()
                sys.exit(1)
                
            # Limit the search terms based on the limit argument
            limit = min(args.limit, len(full_search_terms))
            search_list = full_search_terms[:limit]
            
            print(f"Retrieved {len(full_search_terms)} search terms for date {args.date_list}")
            print(f"Using the first {limit} terms: {search_list}")
            
            # If only date_list is provided without period and value, just print and exit
            if not (args.period and args.value):
                print("Search terms retrieved successfully. Add --period and --value to process data.")
                sys.exit(0)
                
        except Exception as e:
            print(f"Error retrieving search terms: {e}")
            print_usage()
            sys.exit(1)
    
    # Process based on period and value (with or without date_list)
    if args.period and args.value:
        try:
            # Get file paths based on period and value
            initial_file, subsequent_files, start_date_str = get_file_paths(args.period, args.value)
            
            # Validate the files exist
            with open(initial_file, 'r') as f:
                pass  # Just testing if the file can be opened
                
            print(f"Processing {args.period} {args.value}")
            print(f"Initial file: {initial_file}")
            print(f"Start date: {start_date_str}")
            print(f"Number of subsequent files: {len(subsequent_files)}")
            
            if args.date_list:
                print(f"Using {len(search_list)} search terms from date {args.date_list}")
            else:
                print(f"Using {len(default_search_list) if default_search_list else 0} search terms from previous month file")
            
            # Process the data
            combined_df = process_data(initial_file, subsequent_files, start_date_str, args.period, args.value, search_list)
        except FileNotFoundError as e:
            print(f"Error: File not found. {e}")
            print(f"Could not find necessary files for {args.period} {args.value}.")
            print_usage()
            sys.exit(1)
        except Exception as e:
            print(f"Error processing {args.period} {args.value}: {e}")
            print_usage()
            sys.exit(1)
            
    elif args.initial_file and args.subsequent_files and args.start_date and args.period and args.value:
        try:
            # Validate the files exist
            with open(args.initial_file, 'r') as f:
                pass  # Just testing if the file can be opened
                
            print("Using provided file paths")
            print(f"Initial file: {args.initial_file}")
            print(f"Start date: {args.start_date}")
            print(f"Number of subsequent files: {len(args.subsequent_files)}")
            
            if args.date_list:
                print(f"Using {len(search_list)} search terms from date {args.date_list}")
            else:
                print(f"Using {len(default_search_list) if default_search_list else 0} search terms from previous month file")
            
            # Process the data
            combined_df = process_data(
                args.initial_file, 
                args.subsequent_files, 
                args.start_date, 
                args.period, 
                args.value, 
                search_list
            )
        except FileNotFoundError as e:
            print(f"Error: File not found. {e}")
            print("Please check that all file paths are correct and try again.")
            print_usage()
            sys.exit(1)
        except Exception as e:
            print(f"Error processing with provided files: {e}")
            print_usage()
            sys.exit(1)
            
    elif not args.date_list:
        print("Error: Invalid or insufficient arguments provided.")
        print("You must specify one of the following:")
        print_usage()
        sys.exit(1)

        