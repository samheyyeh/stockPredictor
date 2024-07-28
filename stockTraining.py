#!/usr/bin/env python3

import numpy as np
import time as tm
import datetime as dt
import tensorflow as tf
import pandas as pd

from yahoo_fin import stock_info as yf
from sklearn.preprocessing import MinMaxScaler
from collections import deque

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

N_STEPS = 7
LOOKUP_STEPS = [1, 2, 3]
DATE_NOW = tm.strftime('%Y-%m-%d')
DATE_3_YEARS_BACK = (dt.date.today() - dt.timedelta(days=1104)).strftime('%Y-%m-%d')

with open("all_tickers.txt", "r") as f:
    STOCK_TICKERS = f.read().splitlines()

def PrepareData(df, days):
    df['future'] = df['close'].shift(-days)
    last_sequence = np.array(df[['close']].tail(days))
    df.dropna(inplace=True)
    sequence_data = []
    sequences = deque(maxlen=N_STEPS)

    for entry, target in zip(df[['close'] + ['date']].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == N_STEPS:
            sequence_data.append([np.array(sequences), target])

    last_sequence = list([s[:len(['close'])] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)

    X, Y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        Y.append(target)
        
    X = np.array(X)
    Y = np.array(Y)

    return df, last_sequence, X, Y

def GetTrainedModel(x_train, y_train):
    model = Sequential()
    model.add(LSTM(60, return_sequences=True, input_shape=(N_STEPS, len(['close']))))
    model.add(Dropout(0.3))
    model.add(LSTM(120, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(20))
    model.add(Dense(1))

    BATCH_SIZE = 8
    EPOCHS = 80

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)
    model.summary()
    return model

all_predictions = {}

for STOCK in STOCK_TICKERS:
    print(f"Processing stock: {STOCK}")

    init_df = yf.get_data(STOCK, start_date=DATE_3_YEARS_BACK, end_date=DATE_NOW, interval='1d')
    init_df = init_df.drop(['open', 'high', 'low', 'adjclose', 'ticker', 'volume'], axis=1)
    init_df['date'] = init_df.index

    scaler = MinMaxScaler()
    init_df['close'] = scaler.fit_transform(np.expand_dims(init_df['close'].values, axis=1))

    stock_predictions = []
    for step in LOOKUP_STEPS:
        df, last_sequence, x_train, y_train = PrepareData(init_df, step)
        x_train = x_train[:, :, :len(['close'])].astype(np.float32)
        
        model = GetTrainedModel(x_train, y_train)

        last_sequence = last_sequence[-N_STEPS:]
        last_sequence = np.expand_dims(last_sequence, axis=0)
        prediction = model.predict(last_sequence)
        predicted_price = scaler.inverse_transform(prediction)[0][0]

        stock_predictions.append(round(float(predicted_price), 2))

    all_predictions[STOCK] = stock_predictions

model.save('stock_predict_model.h5')

for stock, preds in all_predictions.items():
    print(f"Predictions for {stock}: {preds}")
