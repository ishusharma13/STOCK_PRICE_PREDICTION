import streamlit as st
import plotly.graph_objects as go
from main import fetch_stock_data, fetch_news_sentiment

st.title("📈 Stock Price Prediction & Sentiment Analysis")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")

if st.button("Predict"):
    stock_df = fetch_stock_data(ticker, "2018-01-01", "2024-01-01")
    sentiment_score = fetch_news_sentiment(ticker)
    
    # Plot stock price
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df["Close"], mode="lines", name="Close Price"))
    st.plotly_chart(fig)
    
    # Display Sentiment Score
    st.write(f"News Sentiment Score: {sentiment_score}")
