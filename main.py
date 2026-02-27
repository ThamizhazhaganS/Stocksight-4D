import sys
from data_engine import DataEngine
from model_engine import ModelEngine
from viz_engine import VizEngine
import pandas as pd

def run_analysis_framework(ticker):
    print(f"--- Hybrid LSTM–Monte Carlo Framework: Analyzing {ticker} ---")
    
    # 1. Fetch and process data
    print("[1/4] Fetching market data...")
    engine = DataEngine(ticker)
    try:
        raw_data = engine.fetch_data(period="2y")
        processed_data = engine.prepare_features(raw_data)
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. Train / Load Model
    print("[2/4] Initializing AI Directional Model...")
    model = ModelEngine()
    model.train(processed_data)
    
    # 3. Predict Probabilities for the next 7 days
    print("[3/4] Estimating probabilities...")
    # Get the most recent data point
    latest_row = processed_data.iloc[-1]
    features = latest_row[model.feature_cols].values
    
    probabilities, stats = model.predict_probabilities(features)
    
    print("\nForecast Results (Next 7 Days):")
    for direction, prob in probabilities.items():
        print(f" - {direction}: {prob:.1f}%")
    
    # 4. Visualize
    print("\n[4/4] Generating 4D Visualization...")
    viz = VizEngine(theme='dark')
    viz.plot_4d_forecast(ticker, probabilities)
    print("\nDone! Check the generated image for the forecast.")

if __name__ == "__main__":
    # Allow passing ticker as command line argument
    ticker_input = "AAPL" # Default
    if len(sys.argv) > 1:
        ticker_input = sys.argv[1].upper()
    
    run_analysis_framework(ticker_input)
