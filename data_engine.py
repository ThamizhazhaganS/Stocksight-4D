import yfinance as yf
import pandas as pd
import numpy as np

class DataEngine:
    def __init__(self, ticker):
        self.ticker = ticker

    def fetch_data(self, period="2y", interval="1d"):
        """Fetches historical data from Yahoo Finance."""
        data = yf.download(self.ticker, period=period, interval=interval)
        if data.empty:
            raise ValueError(f"No data found for ticker {self.ticker}")
        
        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        return data

    def prepare_features(self, data):
        """Prepares technical indicators as features."""
        df = data.copy()
        
        # Simple Returns
        df['Returns'] = df['Close'].pct_change()
        
        # Moving Averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        
        # Volatility (Standard Deviation)
        df['Vol_10'] = df['Returns'].rolling(window=10).std()
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9) # Avoid division by zero
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Define Targets for Forecast (e.g., direction in next 7 days)
        # 0: Sideways (→), 1: Bullish (↑), 2: Bearish (↓), 3: Volatile/Reversal (←)
        
        # For training, look ahead 7 days
        df['Target_Next_7d_Return'] = (df['Close'].shift(-7) - df['Close']) / df['Close']
        
        def classify_direction(ret, vol):
            threshold = 0.03 # 3% movement for classification
            if pd.isna(ret): return np.nan
            if ret > threshold: return 1  # Up
            elif ret < -threshold: return 2 # Down
            elif abs(ret) <= threshold and vol < 0.02: return 0 # Sideways
            else: return 3 # Volatile/Uncertain

        df['Direction'] = df.apply(lambda row: classify_direction(row['Target_Next_7d_Return'], row['Vol_10']), axis=1)
        
        # Fill missing values for indicators
        df = df.ffill().dropna(subset=['Returns', 'SMA_20', 'RSI'])
        
        return df

if __name__ == "__main__":
    engine = DataEngine("AAPL")
    raw_data = engine.fetch_data()
    processed_data = engine.prepare_features(raw_data)
    print(processed_data.tail())
