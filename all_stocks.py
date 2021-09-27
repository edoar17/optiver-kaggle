#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 01:32:08 2021

@author: edoar17
"""

import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import lightgbm as lgb
from datetime import datetime
import glob
from joblib import Parallel, delayed

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
# book_train_pathfile_list[:10]

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


############ BIG FN
def complete_feature_processing(parquet_filepath, trade_or_book=''):
    
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
            # df['price_spread1'] = df['ask_price1'].add(-df['bid_price1'])
            # df['price_spread2'] = df['ask_price2'].add(-df['bid_price2'])
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
    
        #functions
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
    ##################################################
    else: 
        print('Need to provide a Trade or Book')
        return
    
    def features_processing(x_df, fe_dict):
        df = x_df.groupby('time_id').agg(fe_dict)
        df.columns = ['_'.join(col) for col in df.columns]
        return df
    
    # Start process
    # Read file
    parquet_df = pd.read_parquet(parquet_filepath)
    # Data wrangling
    pre_features_df = parquet_df.groupby('time_id').apply(features_preprocessing).set_index(['index'], drop=True)

    # Apply Aggs
    processed_features_df = features_processing(pre_features_df, feature_agg_dict).reset_index()
    processed_features_df = processed_features_df.add_prefix(prefix)

    #Clean up
    stock_id = parquet_filepath.split('=')[1]
    x_time_id =  str(prefix + 'time_id')
    processed_features_df['row_id'] = processed_features_df[x_time_id].apply(lambda x: row_id(stock_id, x))
    processed_features_df = processed_features_df.drop([x_time_id], axis=1)
    
    return processed_features_df
    

###### Create final df
# train
train['row_id'] = train['stock_id'].astype(str) + '-' + train['time_id'].astype(str)

def transform_parquets(parquet_pathlist, trade_or_book=''):

    full_dfs = pd.DataFrame()
    
    def for_joblib(path):
        if trade_or_book == 'book':
            new_df = complete_feature_processing(path, trade_or_book='book')
        elif trade_or_book == 'trade': 
            new_df = complete_feature_processing(path, trade_or_book='trade')
        return new_df
    
    # parallel to make it 100x faster
    df = Parallel(n_jobs = -1, verbose = 112)(delayed(for_joblib)(path) for path in parquet_pathlist)
    df = pd.concat(df, ignore_index = True)
    
    return df

### TRADE
# start = datetime.now()
trade_train = transform_parquets(trade_train_pathfile_list, trade_or_book='trade')
trade_test = transform_parquets(trade_test_pathfile_list, trade_or_book='trade')
# end = datetime.now()
# print(end-start) #32 mins
### BOOK
# start = datetime.now()
book_train = transform_parquets(book_train_pathfile_list, trade_or_book='book')
book_test = transform_parquets(book_test_pathfile_list, trade_or_book='book')
# end = datetime.now()
# print(end-start) #1.75hrs

#merge dfs
train_df = book_train.merge(trade_train, on='row_id')
train_df = train.merge(train_df , on='row_id')

test_df = book_test.merge(trade_test, on='row_id')
test_df = test.merge(test_df , on='row_id', how='outer').fillna(0)


#### Feature transform
scaler = MinMaxScaler()
cols_to_scale = [col for col in train_df.columns if col not in ['stock_id', 'time_id', 'row_id', 'target']]
cols_to_scale = [col for col in cols_to_scale if 'volatility' not in col]

train_test_scale = train_df.append(test_df)
train_test_scale[cols_to_scale] = scaler.fit_transform(train_test_scale[cols_to_scale])
train_df[cols_to_scale] = train_test_scale[cols_to_scale].iloc[:-3,]
test_df[cols_to_scale] = train_test_scale[cols_to_scale].iloc[-3:]

#convert data types object ---> category
test_df['time_id'] = test_df['time_id'].astype('category')
test_df['stock_id'] = test_df['stock_id'].astype('category')
train_df['time_id'] = train_df['time_id'].astype('category')
train_df['stock_id'] = train_df['stock_id'].astype('category')


########## MODEL
SEED = 567

params = {
    'objective': 'rmse',
    'boosting_type': 'gbdt',
    'max_depth': -1,
    'max_bin':100,
    'min_data_in_leaf':500,
    'learning_rate': 0.05,
    'subsample': 0.72,
    'subsample_freq': 4,
    'feature_fraction': 0.5,
    'lambda_l1': 0.5,
    'lambda_l2': 1.0,
    'categorical_column':[0],
    'seed':SEED,
    'feature_fraction_seed': SEED,
    'bagging_seed': SEED,
    'drop_seed': SEED,
    'data_random_seed': SEED,
    'n_jobs':-1,
    'verbose': -1}

def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

def feval_rmspe(y_pred, lgb_train):
    y_true = lgb_train.get_label()
    return 'RMSPE', rmspe(y_true, y_pred), False

def train_and_evaluate_lgb(train, test, params):
    # Hyperparameters (just basic)
    
    features = [col for col in train.columns if col not in {"target", "row_id"}]
    y = train['target']
    # Create out of folds array
    oof_predictions = np.zeros(train.shape[0])
    # Create test array to store predictions
    test_predictions = np.zeros(test.shape[0])
    # Create a KFold object
    kfold = KFold(n_splits = 5, random_state = SEED, shuffle = True)
    # Iterate through each fold
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(train)):
        print(f'Training fold {fold + 1}')
        x_train, x_val = train.iloc[trn_ind], train.iloc[val_ind]
        y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]
        # Root mean squared percentage error weights
        train_weights = 1 / np.square(y_train)
        val_weights = 1 / np.square(y_val)
        train_dataset = lgb.Dataset(x_train[features], y_train, weight = train_weights)
        val_dataset = lgb.Dataset(x_val[features], y_val, weight = val_weights)
        model = lgb.train(params = params,
                          num_boost_round=1400,
                          train_set = train_dataset, 
                          valid_sets = [train_dataset, val_dataset], 
                          verbose_eval = 250,
                          early_stopping_rounds=30,
                          feval = feval_rmspe)
        # Add predictions to the out of folds array
        oof_predictions[val_ind] = model.predict(x_val[features])
        # Predict the test set
        test_predictions += model.predict(test[features]) / 5
    rmspe_score = rmspe(y, oof_predictions)
    print(f'Our out of folds RMSPE is {rmspe_score}')
    lgb.plot_importance(model,max_num_features=20)
    # Return test predictions
    return test_predictions

# Traing and evaluate
predictions_lgb = train_and_evaluate_lgb(train_df, test_df, params)
test_df['target'] = predictions_lgb
test_df[['row_id', 'target']].to_csv('submission.csv',index = False)







































