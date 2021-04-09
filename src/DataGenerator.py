import os
import pandas as pd
import numpy as np


def get_data_path(symbol):
    # Return CSV file path given symbol.
    commodity_dir = '../data/commodities'
    currency_dir = '../data/currencies'

    if symbol in ['AUD', 'CNY', 'EUR', 'GBP', 'HKD', 'JPY', 'USD']:
        path = os.path.join(currency_dir, symbol + '_KRW.csv')# _KRW 추가함
    else:
        path = os.path.join(commodity_dir, symbol + '.csv') 

    return path

def merge_data(start_date, end_date, symbols):
    dates = pd.date_range(start_date, end_date)
    df = pd.DataFrame(index=dates)

    if 'Gold' not in symbols:
        symbols.insert(0, 'Gold')

    for symbol in symbols:

        if symbol in ['Gold', 'USD'] :
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
    usd_price = table['USD_Price'].astype(float)

    # TODO:  make features
    gold_price_diff_ratio = ((np.array(gold_price[:-1]) / np.array(gold_price[1:])) -1)*100
    usd_price_diff_ratio = ((np.array(usd_price[:-1]) / np.array(usd_price[1:])) -1)*100
    gold_price = gold_price[1:] 
    usd_price = usd_price[1:] 
    print(gold_price)
    print("==========")
    print(gold_price_diff_ratio)
    print("==========")
    print(usd_price_diff_ratio)
    input_days = 3
    training_sets = list()
    for time in range(len(gold_price)-input_days):
        gold_3days = gold_price_diff_ratio[time:time + input_days]
        usd_3days = usd_price_diff_ratio[time:time + input_days]

        daily_feature = np.concatenate((gold_3days[::-1], usd_3days[::-1]))
        
        training_sets.append(daily_feature)

    print(daily_feature)

    training_x = training_sets[:-10]
    test_x = training_sets[-10:]

    past_price = gold_price[-11:-1]
    target_price = gold_price[-10:]

    return training_x if is_training else (test_x, past_price, target_price)


if __name__ == "__main__":
    start_date = '2010-01-01'
    end_date = '2020-04-18'

    make_features(start_date, end_date, is_training=False)
