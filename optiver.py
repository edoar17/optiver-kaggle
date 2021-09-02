import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq
import numpy as np

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
    xlabel='volatility target').text(0.8, 28, 'Text')



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


book_example = pd.read_parquet('kaggle-download/trade_test.parquet/')





# we load the data from book and train where stock id=0  and time id = 5 
book_0 = pd.read_parquet('kaggle-download/book_train.parquet/stock_id=0')
trade_0 =  pd.read_parquet('kaggle-download/trade_train.parquet/stock_id=0')

###################################################################################

### BOOK_TRAIN is the list of orders per stock ID
### TRADE_TRAIN is the orders that are fulfilled

###################################################################################

# Analyze stock_id=0
# TRAIN
# Price movement
price_plot_0 = sns.lineplot(x='time_id', y='price', data=trade_0).set_title('price of stock_id=0')
trade_0.describe()

price_plot = sns.lineplot(x='time_id', y='price', data=trade_0.iloc[:10000,]).set_title('price of stock_id=0')



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

















