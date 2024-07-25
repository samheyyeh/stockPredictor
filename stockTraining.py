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

import matplotlib.pyplot as plt

# Define constants
N_STEPS = 7
LOOKUP_STEPS = [1, 2, 3]
DATE_NOW = tm.strftime('%Y-%m-%d')
DATE_3_YEARS_BACK = (dt.date.today() - dt.timedelta(days=1104)).strftime('%Y-%m-%d')

# Assume you have a list of stock tickers. Replace this with actual stock tickers.
STOCK_TICKERS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

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

# Store predictions for all stocks
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

    # Optional: Plotting each stock's predictions
    copy_df = init_df.copy()
    y_predicted = model.predict(x_train)
    y_predicted_transformed = scaler.inverse_transform(y_predicted).flatten()

    # Adjust y_predicted_transformed to match the length of copy_df
    if len(y_predicted_transformed) < len(copy_df):
        y_predicted_transformed = np.concatenate([
            np.full(len(copy_df) - len(y_predicted_transformed), np.nan),
            y_predicted_transformed
        ])
    elif len(y_predicted_transformed) > len(copy_df):
        y_predicted_transformed = y_predicted_transformed[-len(copy_df):]

    copy_df[f'predicted_close'] = y_predicted_transformed

    # Convert DATE_NOW to datetime.date object
    date_now = dt.datetime.strptime(DATE_NOW, '%Y-%m-%d').date()

    # Calculate future dates
    future_dates = [date_now + dt.timedelta(days=i) for i in range(3)]

    # Format future dates as strings
    future_dates_str = [d.strftime('%Y-%m-%d') for d in future_dates]

    # Adding new rows to DataFrame
    new_rows = pd.DataFrame({
        'close': stock_predictions,
        'date': future_dates_str,
        'other_col1': [0, 0, 0],
        'other_col2': [0, 0, 0]
    }, index=future_dates_str)

    copy_df = pd.concat([copy_df, new_rows])

    plt.style.use('ggplot')
    plt.figure(figsize=(16,10))
    plt.plot(copy_df['close'][-150:].head(147), label=f'Actual price for {STOCK}')
    plt.plot(copy_df['predicted_close'][-150:].head(147), linewidth=1, linestyle='dashed', label=f'Predicted price for {STOCK}')
    plt.plot(copy_df['close'][-150:].tail(4), label=f'Predicted price for future 3 days')
    plt.xlabel('days')
    plt.ylabel('price')
    plt.title(f'Predictions for {STOCK}')
    plt.legend()
    plt.show()

model.save('stock_predict_model.h5')

for stock, preds in all_predictions.items():
    print(f"Predictions for {stock}: {preds}")
