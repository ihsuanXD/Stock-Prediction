import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Collect Data
ticker = "AAPL"
stock_data = yf.download(ticker, start="2010-01-01", end="2022-03-25")

# Preprocess Data
stock_data = stock_data.dropna()
X = stock_data.drop(["Adj Close"], axis=1)
y = stock_data["Adj Close"]

# Train a Model
model = LinearRegression()
model.fit(X, y)

# Test the Model
predictions = model.predict(X)
mse = mean_squared_error(y, predictions)
rmse = mse**0.5
print(f"RMSE: {rmse}")

# Make Predictions
last_data = stock_data.iloc[-1].drop(["Adj Close"])
prediction = model.predict([last_data])[0]
print(f"Prediction for {ticker}: {prediction}")
