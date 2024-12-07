import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Fetching stock data
def fetch_stock_data(ticker, period='5y'):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data

# Display historical data
def display_data(data):
    print(data.head())
    print(data.describe())

# Calculate moving averages
def add_moving_averages(data):
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()

# Visualize the data with multiple types of graphs
def plot_data(data, ticker):
    plt.figure(figsize=(18, 10))

    # Plot 1: Close Price with Moving Averages
    plt.subplot(2, 3, 1)
    plt.plot(data['Close'], label="Close Price", color="blue")
    plt.plot(data['MA20'], label="20-Day MA", color="red")
    plt.plot(data['MA50'], label="50-Day MA", color="green")
    plt.plot(data['MA200'], label="200-Day MA", color="purple")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"{ticker} - Close Price with Moving Averages")
    plt.legend()

    # Plot 2: Close Price Only
    plt.subplot(2, 3, 2)
    plt.plot(data['Close'], label="Close Price", color="blue")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title(f"{ticker} - Historical Close Price")
    plt.legend()

    # Plot 3: Moving Averages Only
    plt.subplot(2, 3, 4)
    plt.plot(data['MA20'], label="20-Day MA", color="red")
    plt.plot(data['MA50'], label="50-Day MA", color="green")
    plt.plot(data['MA200'], label="200-Day MA", color="purple")
    plt.xlabel("Date")
    plt.ylabel("Moving Averages")
    plt.title(f"{ticker} - Moving Averages")
    plt.legend()

    # Plot 4: Predictions vs Actual Prices (for Linear Regression)
    plt.subplot(2, 3, 5)
    plt.scatter(X_test, y_test, color='blue', label="Actual Price")
    plt.plot(X_test, y_pred, color='red', label="Predicted Price")
    plt.xlabel(r"Days since start")
    plt.ylabel("Close Price")
    plt.title("Stock Price Prediction using Linear Regression")
    plt.legend()

    # Plot 5: Candlestick + Volume Chart
    plt.subplot(2, 3, 5)
    mpf.plot(data[-60:], type='candle', style='charles', volume=True, title=f"{ticker} - Last 60 Days Candlestick + Volume Chart")

    plt.subplots_adjust(wspace=0.3, hspace=0.5)  # Adjust space between plots
    plt.show()

# Prepare data for linear regression
def prepare_data(data):
    data = data[['Close']].dropna()
    data['Date'] = (data.index - data.index.min()).days  # Converting dates to numerical values
    X = data['Date'].values.reshape(-1, 1)
    y = data['Close'].values
    return X, y

# Train the linear regression model and predict
def predict_stock_price(X, y):
    global X_test, y_test, y_pred  # For use in the plot_data function
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate model performance
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

    return model

# Predict future price
def predict_future_price(model, days_ahead):
    future_day = np.array([[days_ahead]])
    future_price = model.predict(future_day)
    return future_price[0]

# Main function
def main():
    ticker = input("Enter the stock ticker symbol (e.g., AAPL for Apple): ").upper()
    data = fetch_stock_data(ticker)
    
    if data.empty:
        print("No data found for the given ticker symbol.")
        return
    
    display_data(data)
    add_moving_averages(data)
    global X, y  # For use in the plot_data function
    X, y = prepare_data(data)
    model = predict_stock_price(X, y)
    plot_data(data, ticker)

    # Predict price 30 days after the latest data
    latest_day = (data.index[-1] - data.index[0]).days
    predicted_price = predict_future_price(model, latest_day + 30)
    print(f"Predicted price for {ticker} 30 days ahead: ${predicted_price:.2f}")

# Run the main function
main()