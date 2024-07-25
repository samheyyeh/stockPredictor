#!/usr/bin/env python3

from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from collections import deque
import datetime as dt
from yahoo_fin import stock_info as yf
import logging

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Define constants
N_STEPS = 7

# Load the model
model = load_model('stock_predict_model.h5')

# Initialize scaler
scaler = MinMaxScaler()

def prepare_data(df, n_steps):
    sequences = deque(maxlen=n_steps)
    X = []

    for entry in df[['close']].values:
        sequences.append(entry)
        if len(sequences) == n_steps:
            X.append(np.array(sequences))

    X = np.array(X)
    return X

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        ticker = data.get('ticker')

        if not ticker:
            return jsonify({'error': 'No ticker symbol provided'}), 400

        logging.debug(f"Received ticker: {ticker}")

        # Get the stock data from Yahoo Finance
        end_date = dt.datetime.now().strftime('%Y-%m-%d')
        start_date = (dt.datetime.now() - dt.timedelta(days=1104)).strftime('%Y-%m-%d')
        df = yf.get_data(ticker, start_date=start_date, end_date=end_date, interval='1d')

        logging.debug(f"Fetched data for {ticker}: {df.head()}")

        if 'close' not in df.columns:
            return jsonify({'error': 'Missing "close" column in the stock data'}), 400

        # Use only the 'close' column
        df = df[['close']]
        df['close'] = scaler.fit_transform(np.expand_dims(df['close'].values, axis=1))

        # Prepare data for prediction
        X_new = prepare_data(df, N_STEPS)
        if len(X_new) == 0:
            return jsonify({'error': 'Not enough data to make predictions'}), 400

        logging.debug(f"Prepared data for prediction: {X_new.shape}")

        # Generate dates for predictions
        dates = pd.date_range(start=df.index[-1], periods=len(X_new) + 1)

        # Make predictions
        predictions = model.predict(X_new)
        predictions = scaler.inverse_transform(predictions).flatten()

        logging.debug(f"Predictions: {predictions}")

        # Convert predictions to list of floats
        predictions = predictions.tolist()

        # Prepare the result with dates and predictions
        result = [{'date': date.strftime('%Y-%m-%d'), 'predicted_close': float(price)}
                  for date, price in zip(dates, predictions)]

        return jsonify({'predictions': result})

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=4444)
