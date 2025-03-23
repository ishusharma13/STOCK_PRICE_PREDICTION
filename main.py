import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import datetime
import plotly.graph_objects as go

def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Example: Fetch Apple (AAPL) stock data from 2018 to 2024
stock_df = fetch_stock_data('AAPL', '2018-01-01', '2024-01-01')
print(stock_df.head())

# Normalize data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(stock_df['Close'].values.reshape(-1,1))

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(scaled_data, seq_length)

# Split into training and testing
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Reshape for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build LSTM model
model = Sequential([




    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32)


# Predict on test data
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Plot results
plt.figure(figsize=(14,6))
plt.plot(stock_df.index[split+seq_length:], stock_df['Close'][split+seq_length:], label='Actual Price')
plt.plot(stock_df.index[split+seq_length:], predictions, label='Predicted Price', linestyle='dashed')
plt.legend()
plt.show()

def fetch_news_sentiment(ticker):
    url = f"https://www.marketwatch.com/investing/stock/{ticker}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    headlines = [h.text for h in soup.find_all('h3')][:10]  # Extract top 10 headlines

    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    
    sentiment_scores = [sia.polarity_scores(headline)["compound"] for headline in headlines]
    return np.mean(sentiment_scores) if sentiment_scores else 0

sentiment_score = fetch_news_sentiment("AAPL")
print("News Sentiment Score:", sentiment_score)
