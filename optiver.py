import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import lightgbm as lgbm
import datetime

# Download Data

train = pd.read_csv('kaggle-download/train.csv')
test = pd.read_csv('kaggle-download/test.csv')
sample = pd.read_csv('kaggle-download/sample_submission.csv')

train.info()
train.describe() #Check statistics

train['stock_id'].value_counts().sort_index() # Rows per stock_id
sns.distplot(train['stock_id']).set_title('density of stock_ids')

train['time_id'].value_counts().sort_values() # Rows per time_id

# Distribution of Target
sns.distplot(train['target'], color = 'b', label = 'target distribution').set(
    title = 'Distribution of target volatility',
    xlabel='volatility target')

def log_return(series):
    return np.log(series).diff()

def pct_change(series):
    return series.pct_change()

def realized_volatility(series):
    return np.sqrt(np.sum(series**2))



# Visualize most volatile stocks
vol = train.groupby('stock_id')['target'].apply(realized_volatility)
vol = vol.to_frame().reset_index()
vol2 = train.groupby('stock_id')['target'].mean()
vol2 = vol2.to_frame().reset_index()
sns.set_style('darkgrid')
sns.set_context("paper")
sns.scatterplot(x='stock_id', y='target', data=vol).set_title('volatility of "target" per stock_id')
sns.scatterplot(x='stock_id', y='target', data=vol2).set_title('mean "target" per stock_id')


vol_stocks = train.nlargest(1000, 'target').sort_values('time_id')
sns.histplot(x='stock_id', data=vol_stocks, bins=50).set_title('frequency of stock_id within 1000 largest volatilities')


# Volatility clustering, times with most volatility
vol_clust = train.nlargest(1000, 'target')
vol_clust.value_counts('time_id') #Proves volatility clustering
sns.histplot(x='time_id', data=vol_clust, bins=50).set_title('frequency of time_id within 1000 largest volatilities')

#Some stocks are more volatility as well as theres times that are more volatile than others.
#Maybe the volatility of the change in volatility can work as a reference.


# we load the data from book and train where stock id=0  and time id = 5 
book_0 = pd.read_parquet('kaggle-download/book_train.parquet/stock_id=0')
trade_0 =  pd.read_parquet('kaggle-download/trade_train.parquet/stock_id=0')

###################################################################################

### BOOK_TRAIN is the list of orders per stock ID
### TRADE_TRAIN is the orders that are fulfilled

###################################################################################


def row_id(stock_id, time_id):
    return f'{int(stock_id)}-{int(time_id)}'


def closest(df, window, price_column):
    ### Algortihm to calculate past [window] seconds_in_buckets' log return
    #find which price to minus
    dif_indexes = []
    idx = 0
    series = df['seconds_in_bucket']
    for x in series:
        # print(x)
        # print(window)
        To_Find = x-window
        # print(To_Find)
        if To_Find>0: 
            if To_Find==series[idx]:
                dif_indexes.append(idx)
            elif abs(series[idx] - To_Find) < abs(series[idx+1]-To_Find):
                dif_indexes.append(idx)
            elif abs(series[idx+1] - To_Find) < abs(series[idx+2]-To_Find):
                dif_indexes.append(idx+1)
                idx += 1
            elif abs(series[idx+2] - To_Find) < abs(series[idx+3]-To_Find):
                dif_indexes.append(idx+2)
                idx += 2
            elif abs(series[idx+3] - To_Find) < abs(series[idx+4]-To_Find):
                dif_indexes.append(idx+3)
                idx += 3
            else: 
                dif_indexes.append(idx+4)
                idx += 4
    # print(len(dif_indexes))
    
    indexes_df = df[df['seconds_in_bucket']>window]
    # print(indexes_df)
    # print(len(indexes_df))
    indexes_df = indexes_df.assign(index_to_minus = dif_indexes)
    # print(indexes_df)    
    
    df_with_indexes = df.merge(indexes_df, how='left')
    df_with_indexes['price_to_minus'] = df_with_indexes['index_to_minus'].apply(lambda x: np.nan if pd.isnull(x) else df[price_column][x])
    
    #calculate window return
    new_col_name = str(price_column) + '_' + str(window)# + '_log_return' # window = 100
    df_with_indexes[new_col_name] = np.log(df_with_indexes[price_column]) - np.log(df_with_indexes['price_to_minus'])
    return df_with_indexes[[new_col_name]] #windowed return

## BOOK
def book_calcs(df):
    # size
    df['ask_size'] = df['ask_size1'].add(df['ask_size2'])
    df['bid_size'] = df['bid_size1'].add(df['bid_size2'])
    df['size_spread'] = df['ask_size'].add(-df['bid_size']) #if negative, bid sz > ask sz
    df['d_size_spread'] = abs(df['size_spread']).diff()
    
    # price
    df['ask_price'] = (df['ask_price1']+df['ask_price2'])/2
    df['d_ask_price'] = pct_change(df['ask_price'])
    df['bid_price'] = (df['bid_price1']+df['bid_price2'])/2
    df['d_bid_price'] = pct_change(df['bid_price'])
    
    df['price_spread'] = df['ask_size']-df['bid_price']
    df['d_price_spread'] = pct_change(df[['price_spread']])
    # df['price_spread1'] = df['ask_price1'].add(df['bid_price1'])
    # df['price_spread2'] = df['ask_price2'].add(df['bid_price2'])
    df['bid_price_spread'] = df['bid_price1'] - df['bid_price2']
    df['d_bid_price_spread'] = pct_change(df['bid_price_spread'])
    df['ask_price_spread'] = df['ask_price2'] - df['ask_price1']
    df['d_ask_price_spread'] = pct_change(df['ask_price_spread'])
    
    # wap
    df['wap1'] = ( df['ask_size1']*df['bid_price1'] + df['bid_size1']*df['ask_price1'] )/(df['ask_size1']+df['bid_size1'])
    df['wap2'] = ( df['ask_size2']*df['bid_price2'] + df['bid_size2']*df['ask_price2'] )/(df['ask_size2']+df['bid_size2'])
    # df['wap3'] = ( df['ask_size1']*df['ask_price1'] + df['bid_size1']*df['bid_price1'] )/(df['ask_size1']+df['bid_size1'])
    # df['wap4'] = ( df['ask_size2']*df['ask_price2'] + df['bid_size2']*df['bid_price2'] )/(df['ask_size2']+df['bid_size2']) 
    
    # wap returns
    df['wap1_ret'] = log_return(df['wap1'])
    df['wap2_ret'] = log_return(df['wap2'])
    # df['wap3_ret'] = log_return(df['wap3'])
    # df['wap4_ret'] = log_return(df['wap4'])
    
    # wap vol
    df['wap1_vol'] = realized_volatility(df['wap1_ret'])
    df['wap2_vol'] = realized_volatility(df['wap2_ret'])
    # df['wap3_vol'] = realized_volatility(df['wap3_ret'])
    # df['wap4_vol'] = realized_volatility(df['wap4_ret'])
    
    #Cumulative returns
    windowed_returns = closest(df.reset_index(), 100, 'wap1')
    df = df.reset_index().merge(windowed_returns, left_index=True, right_index=True)
    
    windowed_returns = closest(df.reset_index(), 200, 'wap1')
    df = df.merge(windowed_returns, left_index=True, right_index=True)
    
    windowed_returns = closest(df.reset_index(), 300, 'wap1')
    df = df.merge(windowed_returns, left_index=True, right_index=True)
    
    windowed_returns = closest(df.reset_index(), 400, 'wap1')
    df = df.merge(windowed_returns, left_index=True, right_index=True)
    
    windowed_returns = closest(df.reset_index(), 500, 'wap1')
    df = df.merge(windowed_returns, left_index=True, right_index=True)

    # print(windowed_returns)
    # print(df)
    return df

start = datetime.datetime.now()
book_example = book_0.groupby('time_id').apply(book_calcs)
end = datetime.datetime.now()
print(end-start)
book_example = book_example.set_index(['index'], drop=True)
# book_example.columns
a = book_example.value_counts(['time_id']).to_frame(name='count')
print(a)


def book_features_processing(book_df, columns_to_keep):
    df = book_df[columns_to_keep_book]
    df = df.groupby('time_id').agg(['min', 'max', 'mean', 'std', 'count'])
    return df

columns_to_keep_book = ['time_id', 'size_spread', 'd_size_spread', 
                   'ask_price', 'bid_price', 'd_ask_price', 'd_bid_price', 
                   'price_spread', 'd_price_spread',
                   'bid_price_spread', 'd_bid_price_spread', 
                   'ask_price_spread', 'd_ask_price_spread',
                   'wap1_ret', 'wap2_ret',
                   'wap1_100', 'wap1_200', 'wap1_300', 'wap1_400', 'wap1_500']
features_book = book_features_processing(book_example, columns_to_keep=columns_to_keep_book)
features_book = features_book.rename_axis(['feature', 'stats'], axis='columns')
# features_book['freq_time_id'] = features_book.merge(a, on='time_id')
desc = features_book.describe()

book_05 = book_0[book_0['time_id']==5]
book_05 = book_05.groupby('time_id').apply(book_calcs)
book_05 = book_05.set_index(['index'], drop=True)

### TRADE 
trade_05 = trade_0[trade_0['time_id']==5]
        
def trade_calcs(df):
    df['price_returns'] = log_return(df['price'])
    df['d_size'] = df['size'].diff()
    df['size_per_order'] = df['size']/df['order_count']
    df['d_order_count'] = df['order_count'].diff()
    # df = df.dropna()
    
    # Cumulative returns
    windowed_returns = closest(df.reset_index(), 100, 'price')
    df = df.reset_index().merge(windowed_returns, left_index=True, right_index=True)
    
    windowed_returns = closest(df.reset_index(), 200, 'price')
    df = df.merge(windowed_returns, left_index=True, right_index=True)
    
    windowed_returns = closest(df.reset_index(), 300, 'price')
    df = df.merge(windowed_returns, left_index=True, right_index=True)
    
    windowed_returns = closest(df.reset_index(), 400, 'price')
    df = df.merge(windowed_returns, left_index=True, right_index=True)
    
    windowed_returns = closest(df.reset_index(), 500, 'price')
    df = df.merge(windowed_returns, left_index=True, right_index=True)
    
    #Realized_volatility
    df['price_vol'] = realized_volatility(df.price_returns)
    return df

start = datetime.datetime.now()
trade_example = trade_0.groupby('time_id').apply(trade_calcs)
end = datetime.datetime.now()
print(end-start)

#N of trades per time id
a = trade_example.value_counts('time_id').to_frame(name='count')
print(a)
# sns.histplot(trade_example['time_id'], bins=5000).set_title('frequency of time_id in trade of stock_id=0')
# trade_example['time_id'].nunique()

def trade_features_processing(trade_df, columns_to_keep):
    df = trade_df[columns_to_keep]
    df = df.groupby('time_id').agg(['min', 'max', 'mean', 'std', 'count'])
    return df

columns_to_keep_trade = ['size', 'd_size',
    'order_count', 'd_order_count', 'size_per_order',
    'price_100', 'price_200', 'price_300', 'price_400', 'price_500']
features_trade = trade_features_processing(trade_example, columns_to_keep_trade)
features_trade = features_trade.rename_axis(['feature', 'stats'], axis='columns')

#### JOIN BOOK AND TRADE FEATURES
full_features = features_trade.merge(features_book, left_index=True, right_index=True)


### Regression lots of trades








## Join calculated volatilities with row ids
trade_vol = trade_example.drop_duplicates('time_id').filter(regex=r'(vol|time)')
book_vol = book_example.drop_duplicates('time_id').filter(regex=r'(vol|time)')


stock_0_train = trade_vol.merge(book_vol, how='left', on='time_id')
stock_0_train = stock_0_vol.merge(train[:3830][['time_id', 'target']], how='left', on='time_id')
stock_0_train['row_id'] = stock_0_train['time_id'].apply(lambda x: f'{0}-{x}')
#rearrange columns
cols = stock_0_train.columns.tolist()
cols = cols[-1:] + cols[:-1]
stock_0_train = stock_0_train[cols].drop(['time_id', 'vwap_vol'], axis=1)


test_book = pd.read_parquet('kaggle-download/book_test.parquet/stock_id=0')
test_trade = pd.read_parquet('kaggle-download/trade_test.parquet/stock_id=0')




#### TRAINING MODEL ####
X = stock_0_train.drop(['target', 'row_id'], axis=1)
Y = stock_0_train['target']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

d_train = lgbm.Dataset(X_train, label=y_train)
lgbm_params = {'learning_rate': 0.05,
               'boosting_type': 'gbdt',
               # 'objective': 'binary',
               'metrics': ['rmspe'],
               'num_leaves': 100,
               'max_depth': 10}

start = datetime.datetime.now()
clf = lgbm.train(lgbm_params, train_set=d_train, num_boost_round=50)
stop = datetime.datetime.now()
execution_time_lgbm = stop-start
print('LGBM execution time is: ', execution_time_lgbm)

y_pred_lgbm = clf.predict(X_test)

rmspe = (np.sqrt(np.mean(np.square((y_test - y_pred_lgbm) / y_test))))
rmspe
def rmspe(y_true, y_pred):
    return  (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))














### TUTO

def realized_volatility_per_time_id(file_path, prediction_column_name):
    df_book_data = pd.read_parquet(file_path)
    
    df_book_data['wap'] = (df_book_data['bid_price1']*df_book_data['ask_size1'] + df_book_data['ask_price1']*df_book_data['bid_size1'])/(df_book_data['bid_size1']+df_book_data['ask_size1'])
    df_book_data['log_return'] = df_book_data.groupby(['time_id'])['wap'].apply(log_return)
    df_book_data = df_book_data[~df_book_data['log_return'].isnull()]
    
    df_realized_vol_per_stock = pd.DataFrame(df_book_data.groupby(['time_id'])['log_return'].agg(realized_volatility)).reset_index()
    df_realized_vol_per_stock = df_realized_vol_per_stock.rename(columns = {'log_return': prediction_column_name})
    
    stock_id = file_path.split('=')[1]
    df_realized_vol_per_stock['row_id'] = df_realized_vol_per_stock['time_id'].apply(lambda x:f'{stock_id}-{x}')
    
    return df_realized_vol_per_stock[['row_id', prediction_column_name]]


c = realized_volatility_per_time_id('kaggle-download/book_train.parquet/stock_id=60','pred')


# trade_example = pd.read_parquet('kaggle-download/trade_test.parquet/')
# book_example = pd.read_parquet('kaggle-download/book_test.parquet/')

















