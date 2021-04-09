import os
import pandas as pd
import numpy as np


def get_data_path(symbol):
    # Return CSV file path given symbol.
    commodity_dir = '../data/commodities'
    currency_dir = '../data/currencies'

    if symbol in ['AUD', 'CNY', 'EUR', 'GBP', 'HKD', 'JPY', 'USD']:
        path = os.path.join(currency_dir, symbol + '.csv')
    else:
        path = os.path.join(commodity_dir, symbol + '.csv')

    return path

def merge_data(start_date, end_date, symbols):
    dates = pd.date_range(start_date, end_date)
    df = pd.DataFrame(index=dates)

    if 'Gold' not in symbols:
        symbols.insert(0, 'Gold')

    for symbol in symbols:

        if symbol == 'Gold' :
            df_temp = pd.read_csv(get_data_path(symbol), index_col="Date", parse_dates=True, na_values=['nan'])
            df_temp = df_temp.applymap(lambda x : x.replace(',', ''))
            df_temp.columns = [symbol + '_' + col for col in df_temp.columns]  # rename columns
            df = df.join(df_temp)
            continue

        df_temp = pd.read_csv(get_data_path(symbol), index_col="Date", parse_dates=True, na_values=['nan'])
        df_temp.columns = [symbol + '_' + col for col in df_temp.columns]  # rename columns
        df = df.join(df_temp)

    return df


def make_features(start_date, end_date, is_training):
    # TODO: select symbols
    # commodity : BrentOil, Copper, CrudeOil, Gasoline, Gold, NaturalGas, Platinum, Silver
    # currency : AUD, CNY, EUR, GBP, HKD, JPY, USD
    table = merge_data(start_date, end_date, symbols=['Gold', 'USD'])

    # TODO: cleaning or filling missing value
    table.dropna(inplace=True)

    # TODO:  select columns to use
    gold_price = table['Gold_Price'].astype(float)

    # TODO:  make features
    gold_diff = np.diff(gold_price)
    gold_price = gold_price[1:] 


    input_days = 3
    training_sets = list()
    for time in range(len(gold_price)-input_days):
        diff = gold_diff[time:time + input_days]
        price = gold_price[time:time + input_days]

        daily_feature = np.concatenate((diff[::-1], price))
        
        training_sets.append(daily_feature)


    training_x = training_sets[:-10]
    test_x = training_sets[-10:]

    past_price = gold_price[-11:-1]
    target_price = gold_price[-10:]

    return training_x if is_training else (test_x, past_price, target_price)


if __name__ == "__main__":
    start_date = '2010-01-01'
    end_date = '2020-04-18'

    make_features(start_date, end_date, is_training=False)
