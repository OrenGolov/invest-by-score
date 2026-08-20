import yfinance as yf

ticker = "AAPL"
stock = yf.Ticker(ticker)
data = stock.history(period="6mo")

print(data.tail())