import pandas as pd


pd.set_option('future.no_silent_downcasting', True)

def process_orders(df):
    """
    Функція process_orders додає нове значення стовпця Orders, якщо SFR не 0, а Orders 0.
    Нове значення є 70% від середнього по всіх інших Orders для цього рядка.
    
    :param df: DataFrame, що містить стовпці з SFR
    :return: DataFrame з новими значеннями стовпця Orders
    """
    sfr_columns = [col for col in df.columns if 'SFR' in col and col != 'Search_TermSFR']
    
    # Для кожного SFR стовпця знаходимо відповідний Orders стовпець
    for sfr_col in sfr_columns:
        # Отримуємо дату з назви стовпця SFR (припускаємо формат 'SFR YYYY-MM-DD')
        date = sfr_col.split('SFR ')[1]
        
        # Формуємо назву відповідного стовпця Orders
        orders_col = f'Orders {date}'
        
        # Перевіряємо чи існує такий стовпець Orders
        if orders_col in df.columns:
            # Отримуємо всі стовпці Orders, крім поточного
            all_orders_columns = [col for col in df.columns if 'Orders' in col and col != orders_col]
            
            # Для кожного рядка
            for idx in df.index:
                # Якщо SFR не 0, а Orders 0
                if df.at[idx, sfr_col] != 0 and df.at[idx, orders_col] == 0:
                    # Рахуємо середнє по всіх інших Orders для цього рядка
                    other_orders = df.loc[idx, all_orders_columns]
                    non_zero_orders = other_orders[other_orders != 0]
                    
                    # Рахуємо середнє по ненульових Orders, якщо такі є
                    if len(non_zero_orders) > 0:
                        mean_orders = non_zero_orders.mean()
                        # Присвоюємо нове значення
                        df.at[idx, orders_col] = round(mean_orders * 0.7)
    return df


def calculate_monthly_metrics(combined_df):
    """
    Функція calculate_monthly_metrics приймає DataFrame з даними за кілька місяців
    і обчислює загальні значення за місяць по стовпцях "Orders" та "Clicks".
    
    :param combined_df: DataFrame з даними за кілька місяців
    :return: DataFrame з новими стовпцями "Monthly Orders" та "Monthly Clicks"
    """
    orders_columns = [col for col in combined_df.columns if 'Orders' in col]
    
    # Створити новий стовпець "Monthly Orders" як суму значень стовпців "Orders" в кожному рядку
    combined_df['Monthly Orders'] = combined_df[orders_columns].sum(axis=1)
    
    # Створити список стовпців, які містять "Clicks"
    clicks_columns = [col for col in combined_df.columns if 'Clicks' in col]
    
    # Створити новий стовпець "Monthly Clicks" як суму значень стовпців "Clicks" в кожному рядку
    combined_df['Monthly Clicks'] = combined_df[clicks_columns].sum(axis=1)
    
    # Створити новий стовпець "Average Orders" як середнє значення стовпців "Orders" в кожному рядку
    combined_df['Average_Orders'] = combined_df[orders_columns].mean(axis=1).round().astype(int)
    
    # Створити новий стовпець "Average Clicks" як середнє значення стовпців "Clicks" в кожному рядку
    combined_df['Average_Clicks'] = combined_df[clicks_columns].mean(axis=1).round().astype(int)

    return combined_df


def calculate_week_metrics(combined_df):
    """
    Функція calculate_week_metrics приймає DataFrame з даними за кілька тижнів
    і обчислює загальні значення за тиждень по стовпцях "Orders" та "Clicks".
    
    :param combined_df: DataFrame з даними за кілька тижнів
    :return: DataFrame з новими стовпцями "Weekly Orders" та "Weekly Clicks"
    """
    orders_columns = [col for col in combined_df.columns if 'Orders' in col]
    
    # Створити новий стовпець "Monthly Orders" як суму значень стовпців "Orders" в кожному рядку
    combined_df['Weekly Orders'] = combined_df[orders_columns].sum(axis=1)
    
    # Створити список стовпців, які містять "Clicks"
    clicks_columns = [col for col in combined_df.columns if 'Clicks' in col]
    
    # Створити новий стовпець "Monthly Clicks" як суму значень стовпців "Clicks" в кожному рядку
    combined_df['Weekly Clicks'] = combined_df[clicks_columns].sum(axis=1)
    
    return combined_df

