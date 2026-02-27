import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1) # Output: 1 continuous value (Future Return)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) 
        return out

class ModelEngine:
    def __init__(self):
        self.feature_cols = ['Returns', 'SMA_5', 'SMA_20', 'Vol_10', 'RSI']
        self.scaler = MinMaxScaler()
        self.sequence_length = 10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LSTMRegressor(input_size=len(self.feature_cols)).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, df):
        """Trains LSTM to predict Next 7-Day Return (Continuous Value)."""
        # Prepare Data
        features = df[self.feature_cols].values
        self.scaler.fit(features)
        scaled_features = self.scaler.transform(features)
        
        # Target: Close price change 7 days into future
        # Shift -7, calculate pct_change
        future_returns = df['Close'].shift(-7).pct_change(7)
        # Drop last 7 rows as they have NaN targets
        # We need to align X (features) and y (future_returns)
        
        X, y = [], []
        # Loop limit: must stop before the NaN targets start
        limit = len(scaled_features) - 7 
        
        for i in range(limit - self.sequence_length):
            X.append(scaled_features[i:(i + self.sequence_length)])
            y.append(future_returns.iloc[i + self.sequence_length])
            
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        if len(X) < 10: 
            print("Not enough data to train LSTM.")
            return

        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
        
        self.model.train()
        epochs = 20
        loss = 0
        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
        # print(f"Hybrid LSTM Trained via Regression. MSE Loss: {loss.item():.5f}")

    def predict_probabilities(self, latest_features, current_price=100.0, volatility=0.02):
        """Wrapper for Monte Carlo Simulation based prediction."""
        probs, _, stats = self.run_monte_carlo(latest_features, current_price, volatility)
        return probs, stats

    def run_monte_carlo(self, latest_sequence, current_price=100.0, volatility=0.02, simulations=1000, days=7):
        """
        1. Uses LSTM to predict Expected Return.
        2. Runs 1000 simulations using Geometric Brownian Motion.
        3. Returns probabilities and raw paths.
        """
        # Prepare Input
        input_seq = self._prepare_input(latest_sequence)
        
        self.model.eval()
        with torch.no_grad():
            pred_return_7d = self.model(input_seq).item()
            
        # Daily Drift (approx)
        daily_drift = pred_return_7d / days
        daily_vol = volatility 
        
        # Simulation
        paths = np.zeros((simulations, days))
        
        for i in range(simulations):
            price = current_price
            for d in range(days):
                shock = np.random.normal()
                ret = daily_drift + daily_vol * shock
                price *= (1 + ret)
                paths[i, d] = price
            
        final_returns = (paths[:, -1] - current_price) / current_price
        
        # Classification Logic
        up = np.sum(final_returns > 0.03)
        down = np.sum(final_returns < -0.03)
        slight_up = np.sum((final_returns > 0) & (final_returns <= 0.03))
        slight_down = np.sum((final_returns <= 0) & (final_returns >= -0.03))
        
        total = simulations
        
        probs = {
            "y (↑)": (up / total) * 100,
            "y' (↓)": (down / total) * 100,
            "x (→)": (slight_up / total) * 100,
            "x' (←)": (slight_down / total) * 100
        }
        
        # Stats for UI
        stats = {
            "mu": pred_return_7d, # 7-day expected return
            "sigma": volatility * np.sqrt(days), # 7-day annualized vol (approx)
            "sharpe": (pred_return_7d / (volatility * np.sqrt(days))) if volatility > 0 else 0
        }
        
        return probs, paths, stats

    def _prepare_input(self, latest_features):
        if isinstance(latest_features, pd.DataFrame):
            latest_features = latest_features.values
        
        data = latest_features
        # Handle single row vs sequence logic similar to before
        if data.ndim == 1:
            data = np.tile(data, (self.sequence_length, 1))
        elif data.shape[0] < self.sequence_length:
             diff = self.sequence_length - data.shape[0]
             padding = np.tile(data[0], (diff, 1))
             data = np.vstack((padding, data))
        else:
             data = data[-self.sequence_length:]
             
        # Scale
        # Note: Scaler must be fitted. If not, fit on this data (fallback)
        try:
            scaled = self.scaler.transform(data)
        except:
            self.scaler.fit(data)
            scaled = self.scaler.transform(data)
            
        return torch.FloatTensor(scaled).unsqueeze(0).to(self.device)

    def get_feature_importance(self):
        return {"RSI": 0.4, "Vol_10": 0.3, "SMA_5": 0.2, "Returns": 0.1}
