import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

def get_data_path(symbol):
    # Return CSV file path given symbol.
    commodity_dir = '../data/commodities'
    currency_dir = '../data/currencies'

    if symbol in ['AUD', 'CNY', 'EUR', 'GBP', 'HKD', 'JPY', 'USD']:
        path = os.path.join(currency_dir, symbol + '_KRW.csv')
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

def simple_moving_average(x, span) :
    sma = []
    for i in range(len(x)) :
        if i <= span-1  :
            sma.append(np.sum(x[0:i+1])/(i+1)) 
        else :
            sma.append(np.sum(x[i-span+1:i+1]/span))

    return sma

def weighted_moving_average(x, span) :
    wma = []
    for i in range(len(x)) :
        if i <= span -1 :
            wma.append(np.sum(np.multiply(x[0:i+1],np.arange(1,i+2)))/np.sum(np.arange(1,i+2)))
        else :
            wma.append(np.sum(np.multiply(x[i-span+1:i+1],np.arange(1,span+1)))/np.sum(np.arange(1,span+1)))

    return wma

def exponential_moving_average(x, span) :
    ema = pd.Series(x).ewm(span=span).mean().to_list()
    
    return ema


def make_feature(x, input_days, span, mode):
    
    price = x
    if mode == "sma":
        ma = simple_moving_average(x, span)
    
    elif mode == "wma":
        ma = weighted_moving_average(x, span)

    elif mode == "ema" : 
        ma = exponential_moving_average(x, span)

    else : 
        raise ValueError("worng argv")

    diff_ma = [0] 
    diff_ma_cont = []

    for i in range(1, len(price)):
        diff_ma.append((price[i] - ma[i-1])/ma[i-1]) 

    for time in range(len(price)-input_days) : 
        diff_ma_cont.append(diff_ma[1:][time: time+input_days])

    return price, ma, diff_ma, diff_ma_cont
    

def make_features(start_date, end_date, mode, input_days, span, is_training):
    # TODO: select symbols
    # commodity : BrentOil, Copper, CrudeOil, Gasoline, Gold, NaturalGas, Platinum, Silver
    # currency : AUD, CNY, EUR, GBP, HKD, JPY, USD
    table = merge_data(start_date, end_date, symbols=['Gold'])

    # TODO: cleaning or filling missing value
    table.dropna(subset=['Gold_Price'], inplace=True)

    # TODO:  select columns to use
    gold_price = table['Gold_Price'].astype(float)

    # TODO:  make features

    gold_price, gold_ma, gold_diff_ma, gold_diff_ma_cont = make_feature(gold_price, input_days, span, mode)
    
    training_sets = list()

    for daily_feature in gold_diff_ma_cont:
        training_sets.append(daily_feature[::-1])

    past_feature = gold_ma[-11:-1]

    training_sets = np.array(training_sets)

    scaler = StandardScaler()
    training_x = scaler.fit_transform(np.array(training_sets[:-10]))

    test_x = scaler.transform(np.array(training_sets[-10:]))


    past_price = gold_price[-11:-1]
    target_price = gold_price[-10:]

    return (training_x, scaler) if is_training else (test_x, past_price, past_feature, target_price, scaler)


if __name__ == "__main__":
    start_date = '2010-01-01'
    end_date = '2020-04-18'
    
    print(make_features(start_date, end_date, mode="wma",input_days=1, span=3, is_training=True)[0].shape)
