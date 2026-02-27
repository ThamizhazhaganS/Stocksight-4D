import pandas as pd
import numpy as np
from data_engine import DataEngine
from model_engine import ModelEngine
from viz_engine import VizEngine
import time

class SimulationEngine:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data_engine = DataEngine(ticker)
        self.model_engine = ModelEngine()
        self.viz_engine = VizEngine(theme='light')

    def run_simulation(self, steps=5):
        """
        Simulates running the model at different points in historical time 
        to see how predictions change.
        """
        print(f"--- Starting Hybrid LSTM–Monte Carlo Framework Simulation for {self.ticker} ---")
        
        # 1. Fetch data
        full_data = self.data_engine.fetch_data(period="2y")
        processed_data = self.data_engine.prepare_features(full_data)
        
        # 2. Initial Training (using all data up to the simulation start)
        # In a real simulation, we'd train on a rolling window, 
        # but for demonstration, we'll train once and predict on the last few days.
        self.model_engine.train(processed_data)
        
        # 3. Step through the last 'steps' days
        last_indices = processed_data.index[-steps:]
        
        for i, current_date in enumerate(last_indices):
            print(f"\n[Step {i+1}/{steps}] Simulating for Date: {current_date.date()}")
            
            # Get features for this specific historical day AND its history context
            # We need the last 15 days ending at 'current_date'
            # Find integer location
            loc = processed_data.index.get_loc(current_date)
            # Slice from loc-15 to loc+1 (inclusive of current)
            # Ideally model needs 10 steps.
            start_loc = max(0, loc - 15)
            sequence_data = processed_data.iloc[start_loc : loc + 1][self.model_engine.feature_cols]
            
            # Predict
            probs, stats = self.model_engine.predict_probabilities(sequence_data)
            
            # Display results
            print(f"Probabilities: {probs}")
            
            # Generate Visualization for this step
            # We'll name them step_1, step_2, etc.
            fig = self.viz_engine.plot_4d_forecast(f"{self.ticker}_Step_{i+1}", probs)
            
            # In a real simulation UI, we'd update a frame here.
            # print(f"Simulation frame saved as forecast_{self.ticker}_Step_{i+1}.png")
            
        print("\n--- Simulation Complete ---")
        print("Generated multiple forecast frames showing how the AI 'sees' the market evolving.")

if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SOL-USD"
    sim = SimulationEngine(ticker)
    sim.run_simulation(steps=3) # Run for last 3 days
