#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 01:32:08 2021

@author: edoar17
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgbm
import datetime
import glob

# Download Data

train = pd.read_csv('kaggle-download/train.csv')
test = pd.read_csv('kaggle-download/test.csv')
sample = pd.read_csv('kaggle-download/sample_submission.csv')

#stock=0
book_0 = pd.read_parquet('kaggle-download/book_train.parquet/stock_id=0')
trade_0 =  pd.read_parquet('kaggle-download/trade_train.parquet/stock_id=0')

book_train_pathfile_list = glob.glob('kaggle-download/book_train.parquet/*')
book_test_pathfile_list = glob.glob('kaggle-download/book_test.parquet/*')
trade_train_pathfile_list = glob.glob('kaggle-download/trade_train.parquet/*')
trade_test_pathfile_list = glob.glob('kaggle-download/trade_test.parquet/*')

book_train_pathfile_list[:10]

def log_return(series):
    return np.log(series).diff()

def pct_change(series):
    return series.pct_change()

def realized_volatility(series):
    return np.sqrt(np.sum(series**2))

def calc_wap1(df):
    wap = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['bid_size1'] + df['ask_size1'])
    return wap

def calc_wap2(df):
    wap = (df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']) / (df['bid_size2'] + df['ask_size2'])
    return wap

def row_id(stock_id, time_id):
    return f'{int(stock_id)}-{int(time_id)}'


def calculate_past_n_log_returns(df, window, price_column_name):
    """Algorithm to calculate past [window] seconds_in_buckets' log return"""
    
    dif_indexes = []
    idx = 0
    series = df['seconds_in_bucket']
    for current_seconds in series:
        To_Find = current_seconds-window
        if To_Find>0: 
            length = len(dif_indexes)
            while len(dif_indexes) == length:
                if (idx+1) <= (len(series)-1):
                    if abs(series[idx] - To_Find) < abs(series[idx+1]-To_Find):   
                        dif_indexes.append(idx)
                        break
                    else: idx += 1
                    print('added one')
                        break
    
    indexes_df = df[df['seconds_in_bucket']>window]
    indexes_df = indexes_df.assign(index_to_minus = dif_indexes)
    # print(indexes_df)    
    
    #merge indexes of prices to minus
    df_with_indexes = df.merge(indexes_df, how='left')
    
    new_col_name = str(price_column_name) + '_' + str(window)
    #df[price]-df[price paired to index]
    df_with_indexes[new_col_name] = df[price_column_name] - df_with_indexes['index_to_minus'].apply(lambda x: np.nan if pd.isnull(x) else df[price_column_name][x])
    return df_with_indexes[[new_col_name]] #windowed return

def trade_features_preprocessing(df):
    df['price_returns'] = log_return(df['price'])
    df['d_size'] = df['size'].diff()
    df['size_per_order'] = df['size']/df['order_count']
    df['d_order_count'] = df['order_count'].diff()
    df['price_amount'] = df['price']*df['size']
    df['d_price_amount'] = df['price_amount'].diff()
    # df = df.dropna()
    print('done')
    # Cumulative returns
    windowed_returns = calculate_past_n_log_returns(df.reset_index(), 100, 'price')
    df = df.reset_index().merge(windowed_returns, left_index=True, right_index=True)
    print('done')
    windowed_returns = calculate_past_n_log_returns(df.reset_index(), 200, 'price')
    df = df.merge(windowed_returns, left_index=True, right_index=True)
    print('done')
    windowed_returns = calculate_past_n_log_returns(df.reset_index(), 300, 'price')
    df = df.merge(windowed_returns, left_index=True, right_index=True)
    print('done')
    # windowed_returns = closest(df.reset_index(), 400, 'price')
    # df = df.merge(windowed_returns, left_index=True, right_index=True)
    
    # windowed_returns = closest(df.reset_index(), 500, 'price')
    # df = df.merge(windowed_returns, left_index=True, right_index=True)
    
    #Realized_volatility
    # df['price_vol'] = realized_volatility(df.price_returns)
    return df

def trade_features_processing(trade_df, fe_dict):
    df = trade_df.groupby('time_id').agg(fe_dict)
    df.columns = ['_'.join(col) for col in df.columns]
    return df

def trade_complete_feature_processing(trade_parquet_filepath, stock_id): 
    parquet_df = pd.read_parquet(trade_parquet_filepath)
    preprocessed_features_df = parquet_df.groupby('time_id').apply(trade_features_preprocessing).set_index(['index'], drop=True)

    feature_agg_dict = {
    'price_returns': [realized_volatility],
    'd_size': [np.max, np.min, np.std],
    'size_per_order': [np.mean, np.std],
    'd_order_count': [np.mean, np.max, np.std],
    'price_amount': [np.max, np.min, np.std, 'count'],#get # of trade entries
    'd_price_amount': [np.max, np.min, np.std]
    }
    
    processed_features_df = trade_features_processing(preprocessed_features_df, feature_agg_dict).reset_index()
    processed_features_df = processed_features_df.add_prefix('trade_')
    processed_features_df['row_id'] = processed_features_df['trade_time_id'].apply( lambda x: row_id(stock_id, x))
    processed_features_df = processed_features_df.drop(['trade_time_id'], axis=1)
    
    return processed_features_df

##### CADA UNO TIENE

start = datetime.datetime.now()
eee = trade_features_preprocessing(book_example)
end = datetime.datetime.now()
print(end-start) #6.5mins

rrr = trade_features_processing(eee, feature_agg_dict)










def complete_feature_processing(parquet_filepath, trade_or_book):
    
    if trade_or_book=='book':
        
        prefix = 'book_'
        feature_agg_dict = {
        'wap1_ret': [realized_volatility],
        'wap2_ret': [realized_volatility],
        'wap1_100': [realized_volatility],
        'wap1_200': [realized_volatility],
        'wap1_300': [realized_volatility],
        'wap1_400': [realized_volatility],
        'wap1_500': [realized_volatility],
        'ret_ask_price':[realized_volatility],
        'ret_bid_price':[realized_volatility],
        'ask_price_spread':[np.max, np.min, np.std],
        'd_ask_price_spread': [np.max, np.min, np.std],
        'bid_price_spread':[np.max, np.min, np.std],
        'd_bid_price_spread': [np.max, np.min, np.std],
        'size_spread': [np.max, np.min, np.std, 'count'], #get # of book entries
        'd_size_spread': [np.max, np.min, np.std],
        }
        
        #functions
        def features_preprocessing(df):
            
            # size
            df['ask_size'] = df['ask_size1'] + df['ask_size2']
            df['bid_size'] = df['bid_size1'] + df['bid_size2']
            df['size_spread'] = abs(df['ask_size'] - df['bid_size'])
            df['d_size_spread'] = df['size_spread'].diff()
            
            # price
            df['ask_price'] = (df['ask_price1']+df['ask_price2'])/2
            df['ret_ask_price'] = log_return(df['ask_price'])
            df['bid_price'] = (df['bid_price1']+df['bid_price2'])/2
            df['ret_bid_price'] = log_return(df['bid_price'])
            
            df['price_spread'] = abs(df['ask_size']-df['bid_price'])
            df['d_price_spread'] = df['price_spread'].diff()
            # df['price_spread1'] = df['ask_price1'].add(df['bid_price1'])
            # df['price_spread2'] = df['ask_price2'].add(df['bid_price2'])
            df['bid_price_spread'] = df['bid_price1'] - df['bid_price2']
            df['d_bid_price_spread'] = df['bid_price_spread'].diff()
            df['ask_price_spread'] = df['ask_price2'] - df['ask_price1']
            df['d_ask_price_spread'] = df['ask_price_spread'].diff()
            
            # wap
            df['wap1'] = calc_wap1(df)
            df['wap2'] = calc_wap2(df)
    
            # wap returns
            df['wap1_ret'] = log_return(df['wap1'])
            df['wap2_ret'] = log_return(df['wap2'])
    
            
            #Cumulative returns
            windowed_returns = calculate_past_n_log_returns(df.reset_index(), 100, 'wap1')
            df = df.reset_index().merge(windowed_returns, left_index=True, right_index=True)
            
            windowed_returns = calculate_past_n_log_returns(df.reset_index(), 200, 'wap1')
            df = df.merge(windowed_returns, left_index=True, right_index=True)
            
            windowed_returns = calculate_past_n_log_returns(df.reset_index(), 300, 'wap1')
            df = df.merge(windowed_returns, left_index=True, right_index=True)
            
            windowed_returns = calculate_past_n_log_returns(df.reset_index(), 400, 'wap1')
            df = df.merge(windowed_returns, left_index=True, right_index=True)
            
            windowed_returns = calculate_past_n_log_returns(df.reset_index(), 500, 'wap1')
            df = df.merge(windowed_returns, left_index=True, right_index=True)
            return df
        
        
#####################################################     
    elif trade_or_book=='trade':
            
        prefix = 'trade_'
        feature_agg_dict = {
        'price_returns': [realized_volatility],
        'd_size': [np.max, np.min, np.std],
        'size_per_order': [np.mean, np.std],
        'd_order_count': [np.mean, np.max, np.std],
        'price_amount': [np.max, np.min, np.std, 'count'],#get # of trade entries
        'd_price_amount': [np.max, np.min, np.std]
        }
    
        def features_preprocessing(df):
           
            df['price_returns'] = log_return(df['price'])
            df['d_size'] = df['size'].diff()
            df['size_per_order'] = df['size']/df['order_count']
            df['d_order_count'] = df['order_count'].diff()
            df['price_amount'] = df['price']*df['size']
            df['d_price_amount'] = df['price_amount'].diff()
            # Cumulative returns
            windowed_returns = calculate_past_n_log_returns(df.reset_index(), 100, 'price')
            df = df.reset_index().merge(windowed_returns, left_index=True, right_index=True)
            # print('done 100')
            windowed_returns = calculate_past_n_log_returns(df.reset_index(), 200, 'price')
            df = df.merge(windowed_returns, left_index=True, right_index=True)
            # print('done 200')
            windowed_returns = calculate_past_n_log_returns(df.reset_index(), 300, 'price')
            df = df.merge(windowed_returns, left_index=True, right_index=True)
            # print('done 300')
            return df
    ############################
    else: 
        print('Need to provide a Trade or Book')
    
    def features_processing(x_df, fe_dict):
        df = x_df.groupby('time_id').agg(fe_dict)
        df.columns = ['_'.join(col) for col in df.columns]
        return df
    
    # Start process
    #read file
    parquet_df = pd.read_parquet(parquet_filepath)
    #data wrangling
    pre_features_df = parquet_df.groupby('time_id').apply(features_preprocessing).set_index(['index'], drop=True)

    # Apply Aggs
    processed_features_df = features_processing(pre_features_df, feature_agg_dict).reset_index()
    processed_features_df = processed_features_df.add_prefix(prefix)

    #Clean up
    stock_id = parquet_filepath.split('=')[1]
    x_time_id =  str(prefix + 'time_id')
    processed_features_df['row_id'] = processed_features_df[x_time_id].apply( lambda x: row_id(stock_id, x))
    processed_features_df = processed_features_df.drop([x_time_id], axis=1)
    
    return processed_features_df
    

#book TESTING
start = datetime.datetime.now()
pq = book_train_pathfile_list[2]
TESTING_BOOK = complete_feature_processing(pq, trade_or_book='book')
end = datetime.datetime.now()
print(end-start) #5.75mins
start = datetime.datetime.now()
pq2 = trade_train_pathfile_list[7]
pq3 = 'kaggle-download/trade_train.parquet/stock_id=0'
TESTING_TRADE = complete_feature_processing(pq3, trade_or_book='trade')
end = datetime.datetime.now()
print(end-start)#1min30

def calculate_past_n_log_returns(df, window, price_column_name):
    """Algorithm to calculate past [window] seconds_in_buckets' log return"""
    
    dif_indexes = []
    idx = 0
    series = df['seconds_in_bucket']
    for current_seconds in series:
        To_Find = current_seconds-window
        if To_Find>0: 
            length = len(dif_indexes)
            while len(dif_indexes) == length:
                if (idx+1) <= (len(series)-1):
                    if abs(series[idx] - To_Find) < abs(series[idx+1]-To_Find):   
                        dif_indexes.append(idx)
                        break
                    else: 
                        idx += 1
                else: 
                    break
    
    indexes_df = df[df['seconds_in_bucket']>window]
    if len(dif_indexes) == len(indexes_df):
        indexes_df = indexes_df.assign(index_to_minus = dif_indexes)
    else: 
        dif_indexes = [np.nan] + dif_indexes
        indexes_df = indexes_df.assign(index_to_minus = dif_indexes)    
    # print(indexes_df)    
    
    #merge indexes of prices to minus
    df_with_indexes = df.merge(indexes_df, how='left')
    
    new_col_name = str(price_column_name) + '_' + str(window)
    #df[price]-df[price paired to index]
    df_with_indexes[new_col_name] = df[price_column_name] - df_with_indexes['index_to_minus'].apply(lambda x: np.nan if pd.isnull(x) else df[price_column_name][x])
    return df_with_indexes[[new_col_name]] #windowed return











