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
                if abs(series[idx] - To_Find) < abs(series[idx+1]-To_Find):   
                    dif_indexes.append(idx)
                    break
                else: idx += 1
    
    indexes_df = df[df['seconds_in_bucket']>window]
    indexes_df = indexes_df.assign(index_to_minus = dif_indexes)
    # print(indexes_df)    
    
    #merge indexes of prices to minus
    df_with_indexes = df.merge(indexes_df, how='left')
    
    new_col_name = str(price_column_name) + '_' + str(window)
    #df[price]-df[price paired to index]
    df_with_indexes[new_col_name] = df[price_column_name] - df_with_indexes['index_to_minus'].apply(lambda x: np.nan if pd.isnull(x) else df[price_column_name][x])
    return df_with_indexes[[new_col_name]] #windowed return


## BOOK
def book_features_preprocessing(df):
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
    # df['wap1'] = ( df['ask_size1']*df['bid_price1'] + df['bid_size1']*df['ask_price1'] )/(df['ask_size1']+df['bid_size1'])
    # df['wap2'] = ( df['ask_size2']*df['bid_price2'] + df['bid_size2']*df['ask_price2'] )/(df['ask_size2']+df['bid_size2'])
    # df['wap3'] = ( df['ask_size1']*df['ask_price1'] + df['bid_size1']*df['bid_price1'] )/(df['ask_size1']+df['bid_size1'])
    # df['wap4'] = ( df['ask_size2']*df['ask_price2'] + df['bid_size2']*df['bid_price2'] )/(df['ask_size2']+df['bid_size2']) 
    
    # wap returns
    df['wap1_ret'] = log_return(df['wap1'])
    df['wap2_ret'] = log_return(df['wap2'])
    # df['wap3_ret'] = log_return(df['wap3'])
    # df['wap4_ret'] = log_return(df['wap4'])
    
    # wap vol
    # df['wap1_vol'] = realized_volatility(df['wap1_ret'])
    # df['wap2_vol'] = realized_volatility(df['wap2_ret'])
    # df['wap3_vol'] = realized_volatility(df['wap3_ret'])
    # df['wap4_vol'] = realized_volatility(df['wap4_ret'])
    
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

    # print(windowed_returns)
    # print(df)
    return df

start = datetime.datetime.now()
book_example = book_0.groupby('time_id').apply(book_features_preprocessing).set_index(['index'], drop=True)
end = datetime.datetime.now()
print(end-start)

def book_features_processing(book_df, fe_dict):
    df = book_df.groupby('time_id').agg(fe_dict)
    df.columns = ['_'.join(col) for col in df.columns]
    return df

features_book = book_features_processing(book_example, fe_dict=feature_agg_dict).reset_index()
www = features_book['time_id'].apply(lambda x: )

def book_complete_feature_processing(book_parquet_filepath, stock_id): 
    parquet_df = pd.read_parquet(book_parquet_filepath)
    preprocessed_features_df = parquet_df.groupby('time_id').apply(book_features_preprocessing).set_index(['index'], drop=True)

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
    
    processed_features_df = book_features_processing(preprocessed_features_df, feature_agg_dict).reset_index()
    processed_features_df = processed_features_df.add_prefix('book_')
    processed_features_df['row_id'] = processed_features_df['book_time_id'].apply( lambda x: row_id(stock_id, x)).drop(['book_time_id'], axis=1)
    
    return processed_features_df

www = book_train_pathfile_list[1]

start = datetime.datetime.now()
qqq = book_complete_feature_processing(www, stock_id=www.split('=')[1])
end = datetime.datetime.now()
print(end-start) #5mins

   
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
    processed_features_df['row_id'] = processed_features_df['time_id'].apply( lambda x: row_id(stock_id, x))
    processed_features_df = processed_features_df.add_prefix('trade_').drop(['trade_time_id'], axis=1)
    
    return processed_features_df

start = datetime.datetime.now()
eee = trade_features_preprocessing(trade_0)
end = datetime.datetime.now()
print(end-start) #6.5mins

rrr = trade_features_processing(eee, feature_agg_dict)




