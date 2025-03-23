# Stock Price Prediction using Machine Learning

## 📌 Project Overview
This project predicts stock prices based on historical data, market trends, and news sentiment analysis using machine learning. The model uses **LSTM (Long Short-Term Memory)** for time series prediction and **NLP (Natural Language Processing)** for sentiment analysis of financial news.

## 🚀 Features
- 📈 Fetches real-time **stock data** using Yahoo Finance API.
- 📰 Scrapes and analyzes **news sentiment** related to the stock.
- 🤖 Uses **LSTM, ARIMA, or XGBoost** for stock price prediction.
- 📊 Provides a **Streamlit dashboard** for visualization.

---

## 🛠️ Tech Stack
- **Programming Language:** Python
- **Libraries:** Pandas, Scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn, Plotly
- **Data Sources:** Yahoo Finance API, Web Scraping (BeautifulSoup)
- **Frameworks:** Streamlit (for Dashboard UI)

---

## 📂 Project Structure
```
├── main.py          # Data fetching, training, and prediction
├── app.py           # Streamlit dashboard for visualization
├── requirements.txt # Required Python packages
├── README.md        # Project documentation
```

---

## 📥 Installation & Setup
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/ishusharma/stock-price-prediction.git
cd stock-price-prediction
```

### **2️⃣ Create a Virtual Environment (Optional but Recommended)**
```sh
python -m venv venv
```
Activate it:
- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

### **3️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```
If you don’t have `requirements.txt`, run:
```sh
pip install numpy pandas scikit-learn tensorflow keras yfinance beautifulsoup4 requests nltk matplotlib seaborn plotly streamlit
```

---

## 📊 Running the Project
### **1️⃣ Train the Stock Price Prediction Model**
```sh
python main.py
```
This will:
- Fetch historical stock data
- Perform sentiment analysis
- Train an LSTM-based prediction model

### **2️⃣ Run the Streamlit Dashboard**
```sh
streamlit run app.py
```
This will launch an interactive web app to visualize stock predictions.

---

## 📌 Example Output
- **Stock Price Prediction Graph** 📉
- **Sentiment Analysis Score** 📰
- **Real-time Data & Forecast** 📊



