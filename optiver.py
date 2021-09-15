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
sns.distplot(train['stock_id'])

train['time_id'].value_counts().sort_index() # Rows per time_id

# Distribution of Target
sns.distplot(train['target'], color = 'b', label = 'target distribution').set(
    title = 'Distribution of target volatility',
    xlabel='volatility target')



# Visualize most volatile stocks
vol = train.groupby('stock_id').mean()
vol_stocks = train.nlargest(1000, 'target').sort_values('time_id')
sns.histplot(x='stock_id', data=vol_stocks, bins=50).set_title(
    'frequency of time_id within 1000 largest volatilities')

# Volatility clustering, times with most volatility
vol_clust = train.nlargest(1000, 'target')
vol_clust.value_counts('time_id') #Proves volatility clustering
sns.histplot(x='time_id', data=vol_clust, bins=50).set_title(
    'frequency of time_id within 1000 largest volatilities')

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

def log_return(series):
    return np.log(series).diff()

def realized_volatility(series):
    return np.sqrt(np.sum(series**2))


def closest(df, window, price_column):
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
            elif abs(series[idx+1] - To_Find) < abs(series[idx+2]-x):
                dif_indexes.append(idx+1)
                idx += 1
            elif abs(series[idx+2] - To_Find) < abs(series[idx+3]-x):
                dif_indexes.append(idx+2)
                idx += 2
            elif abs(series[idx+3] - To_Find) < abs(series[idx+4]-x):
                dif_indexes.append(idx+3)
                idx += 3
    
    # print(len(dif_indexes))
    
    indexes_df = df[df['seconds_in_bucket']>window]
    # print(indexes_df)
    # print(len(indexes_df))
    indexes_df = indexes_df.assign(index_to_minus = dif_indexes)
    # print(indexes_df)    
    
    df_with_indexes = df.merge(indexes_df, how='left')
    df_with_indexes['price_to_minus'] = df_with_indexes['index_to_minus'].apply(lambda x: np.nan if pd.isnull(x) else df[price_column][x])
    
    #calculate window return
    new_col_name = str(price_column) + '_' + str(window) + '_log_return' # window = 100
    df_with_indexes[new_col_name] = np.log(df_with_indexes[price_column]) - np.log(df_with_indexes['price_to_minus'])
    return df_with_indexes[[new_col_name]] #windowed return

## Stock LIQUIDITY
def book_calcs(df):
    # size
    df['ask_size'] = df['ask_size1'].add(df['ask_size2'])
    df['bid_size'] = df['bid_size1'].add(df['bid_size2'])
    df['size_spread'] = df['ask_size'].add(-df['bid_size']) #if negative, bid sz > ask sz
    df['median_size'] = df['ask_size'].add(-df['bid_size'])/2
    # price
    df['ask_price'] = (df['ask_price1']+df['ask_price2'])/2
    df['bid_price'] = (df['bid_price1']+df['bid_price2'])/2
    df['price_spread'] = df['ask_price1'].add(df['bid_price1'])
    df['price_spread2'] = df['ask_price2'].add(df['bid_price2'])
    df['bid_price_spread'] = df['bid_price1'] - df['bid_price2']
    df['ask_price_spread'] = df['ask_price2'] - df['ask_price1']
    # wap
    df['wap1'] = ( df['ask_size1']*df['bid_price1'] + df['bid_size1']*df['ask_price1'] )/(df['ask_size1']+df['bid_size1'])
    df['wap2'] = ( df['ask_size2']*df['bid_price2'] + df['bid_size2']*df['ask_price2'] )/(df['ask_size2']+df['bid_size2'])
    df['wap3'] = ( df['ask_size1']*df['ask_price1'] + df['bid_size1']*df['bid_price1'] )/(df['ask_size1']+df['bid_size1'])
    df['wap4'] = ( df['ask_size2']*df['ask_price2'] + df['bid_size2']*df['bid_price2'] )/(df['ask_size2']+df['bid_size2']) 
    # wap returns
    df['wap1_ret'] = log_return(df['wap1'])
    df['wap2_ret'] = log_return(df['wap2'])
    df['wap3_ret'] = log_return(df['wap3'])
    df['wap4_ret'] = log_return(df['wap4'])
    # wap vol
    df['wap1_vol'] = realized_volatility(df['wap1_ret'])
    df['wap2_vol'] = realized_volatility(df['wap2_ret'])
    df['wap3_vol'] = realized_volatility(df['wap3_ret'])
    df['wap4_vol'] = realized_volatility(df['wap4_ret'])
    
    
    def calculateWindowedReturns(df, price_column, window):
        windowed_returns = pd.DataFrame()
        for i in df['time_id'].unique():
            working_df = df[df['time_id']==i].reset_index()
            rets = closest(working_df, window=window, price_column=price_column)
            if not windowed_returns.empty:
                windowed_returns = windowed_returns.append(rets, ignore_index=True)
            else:
                windowed_returns=rets
        return windowed_returns
        
            
            # working_df = df[df['time_id']==time_id].reset_index()
            # return closest(working_df, window=window, price_column=price_column)
    
    # returns = pd.DataFrame()
    # for i in df['time_id'].unique():
    #     # print(i)
    #     rets = calculateWindowedReturns(df, price_column='wap1', window=100, time_id=i)
    #     # print(i)
    #     if not returns.empty:
    #         returns = returns.append(rets, ignore_index=True)
    #     else:
    #         returns=rets
    df['wap1_200'] = calculateWindowedReturns(df, price_column='wap1', window=200)        
    df['wap1_100'] = calculateWindowedReturns(df, price_column='wap1', window=100)       
    df['wap1_300'] = calculateWindowedReturns(df, price_column='wap1', window=300)
    df['wap1_500'] = calculateWindowedReturns(df, price_column='wap1', window=500)
    
    return df

book_example = book_0.groupby('time_id').apply(book_calcs)

book_062 = book_example[book_example['time_id']==62].reset_index()
calculateWindowedReturns(book_062, price_column='wap1', window=100, time_id=62)
qqq = closest(book_062, window=100, price_column='wap1')

book_05 = book_example[(book_example['time_id']==5) | (book_example['time_id']==11)]

qqq = calculateWindowedReturns(book_05, 'wap1', 100, 11)
qqq = closest(book_05[book_05['time_id']==11].reset_index(), 100, 'wap1')

returns = pd.DataFrame()
type(returns)
del(returns)

for i in book_05['time_id'].unique():
    rets = calculateWindowedReturns(book_05, 'wap1', 100, i)
    # returns.append(rets)
    if not returns.empty:
        returns = returns.append(rets, ignore_index=True)
    else:
        returns=rets
        
closest(book_05, 300, 'wap1')

book_example = book_0.groupby('time_id').apply(book_calcs)

qqq = closest(r, 100, 'price')
qqq = closest(book_05, 0, 'wap1')

book_example = book_0.groupby('time_id').apply(book_calcs)
book_example.columns

### VOLATILITY 
trade_05 = trade_0[trade_0['time_id']==5]
        
def trade_calcs(df):
    df['vwap'] = (df['price']*df['size'])/df['size']
    df['price_returns'] = log_return(df['price'])
    df['vwap_returns'] = log_return(df['vwap'])
    df = df.dropna()
    #get realized_volatility
    df['price_vol'] = realized_volatility(df.price_returns)
    df['vwap_vol'] = realized_volatility(df.vwap_returns)
    return df

trade_example = trade_0.groupby('time_id').apply(trade_calcs).reset_index(drop=True)

#N of trades per time id
a = trade_example.value_counts('time_id')
sns.histplot(a, y='time_id', bins=1000)
# trade_example['time_id'].nunique()

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


def log_return(prices):
    return np.log(prices).diff()

def realized_volatility(logrets):
    return np.sqrt(np.sum(logrets**2))




trade_example = pd.read_parquet('kaggle-download/trade_test.parquet/')
book_example = pd.read_parquet('kaggle-download/book_test.parquet/')

















