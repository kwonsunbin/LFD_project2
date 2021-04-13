from test import test
from training import train
import pandas as pd
import numpy as np
import os
import pickle

dates_set = []

for date in pd.date_range("2015-01-01","2021-04-01",freq="QS").tolist() :
    dates_set.append(('2010-01-01',str(date).split(" ")[0]))

def input_days_and_span_selection() :
    input_days_span_combination = []

    for input_days in range(1, 30) :
        for span in range(1, 30):
          input_days_span_combination.append((input_days, span))

    result = []

    for combination in input_days_span_combination :
        input_days = combination[0]
        span = combination[1]
        
        temp_result = []
        for date in dates_set :
            start_date = date[0]
            end_date = date[1]
            train(start_date, end_date, "sma", input_days, span, 15)
            temp_result.append(test(start_date, end_date, "sma", input_days, span, 15))
        result.append((input_days, span, temp_result))

    pickle.dump(result, open("inputdays&span", 'wb'))

def moving_average_selection() :

    available = ["sma", "wma", "ema"]

    result = {"sma": [], "wma" : [], "ema" : []}
    
    for moving_average in available :
        temp_result = []
        for date in dates_set :
            start_date = date[0]
            end_date = date[1]
            train(start_date, end_date, moving_average, 1, 3, 15)
            temp_result.append(test(start_date, end_date, moving_average, 1, 3, 15))

        result[moving_average] = temp_result

    pickle.dump(result, open("movingAVG", 'wb'))

def n_components_selection() :
    result = {}
    for n in range(1,10):
        temp_result = []
        for date in dates_set :
            start_date = date[0]
            end_date = date[1]
            train(start_date, end_date, "sma", 1, 3, n)
            temp_result.append(test(start_date, end_date, "sma", 1, 3, n))

        result[n] = temp_result

    pickle.dump(result, open("n_components", 'wb'))
  


def main():
    #input_days_and_span_selection()
    moving_average_selection()
    #n_components_selection()

if __name__ == '__main__':
    main()