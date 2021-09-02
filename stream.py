#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 23:53:25 2021

@author: edoar17
"""


# Download Data
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import pyarrow.parquet as pq
import os
import glob

train = pd.read_csv('kaggle-download/train.csv')
test = pd.read_csv('kaggle-download/test.csv')
sample = pd.read_csv('kaggle-download/sample_submission.csv')
list_order_book_file_train = glob.glob('kaggle-download/book_train.parquet/*')

book_0 = pd.read_parquet('kaggle-download/book_train.parquet/stock_id=0')
trade_0 =  pd.read_parquet('kaggle-download/trade_train.parquet/stock_id=0')

a = book_0.head(10000)
a = a[a['time_id']==5]
b = trade_0.head(10000)
b = b[b['time_id']==5]

#######Animated

plt.style.use('fivethirtyeight')
x_data = a['seconds_in_bucket']
y_data = a['bid_price1']
y2_data = a['ask_price1']

x_vals = []
y_vals = []
y2_vals = []
spread = []
vwap = []


def animate(i):
    #Bid price, ask price, spread, vwap
    idx = 0
    for i in x_data:
        if len(x_vals)==idx:
            x_vals.append(i)
            y_vals.append(y_data[idx])
            y2_vals.append(y2_data[idx])
            spread.append((y2_vals[idx]-y_vals[idx])+1)
            vwap.append(a['vwap'][idx])
            break
        else: idx += 1   
    
    plt.cla()
    plt.plot(x_vals, y_vals, label='bid price')
    plt.plot(x_vals, y2_vals, label='ask price')
    plt.plot(x_vals, spread, label='spread')
    # plt.plot(x_vals, vwap, label='vwap')
    plt.plot(b['seconds_in_bucket'], b['price'])
    plt.scatter(b['seconds_in_bucket'], b['price'])
    plt.legend(loc='upper left')
    
ani = FuncAnimation(plt.gcf(), animate, interval=100)
plt.tight_layout()
%matplotlib qt
plt.show()


# Add VWAP
a
a['vwap'] = (a['bid_price1']*a['ask_size1'] + a['ask_price1']*a['bid_size1']) / (a['bid_size1'] + a['ask_size1'])
b['denom'] = b['size'].cumsum()
b['numer'] = (b['price']*b['size']).cumsum()
b['vwap'] = b['numer']/b['denom']
b = b.drop(['denom', 'numer'], axis=1)


p = plt.plot(b['seconds_in_bucket'], b['vwap'])
plt.show()


c = a.merge(b, how='right', on='seconds_in_bucket')
c = c.drop(['time_id_x', 'vwap_x', 'time_id_y'], axis=1)















