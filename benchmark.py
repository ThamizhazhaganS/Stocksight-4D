import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from io import StringIO
import time

# Import project components
from data_engine import DataEngine
from model_engine import ModelEngine

def run_benchmark(ticker="SOL-USD"):
    print(f"--- Running Accuracy Benchmark for {ticker} ---")
    
    # 1. Fetch & Prepare Data
    print("[1/5] Fetching 2 years of data...")
    data_engine = DataEngine(ticker)
    try:
        raw_data = data_engine.fetch_data(period="2y")
        df = data_engine.prepare_features(raw_data)
        
        # Drop NaN targets for training/testing (last 7 days won't have valid targets)
        df = df.dropna(subset=['Direction'])
        
        # Split into Train/Test (Time-based split, not random)
        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size].copy()
        test_df = df.iloc[train_size:].copy()
        
        print(f"Data Points: {len(df)} | Train: {len(train_df)} | Test: {len(test_df)}")
        
        feature_cols = ['Returns', 'SMA_5', 'SMA_20', 'Vol_10', 'RSI']
        target_col = 'Direction' # 0, 1, 2, 3
        
        results = []

        # --- MODEL 1: LSTM (Proposed) ---
        print("\n[2/5] Evaluating LSTM (Proposed Model)...")
        start_time = time.time()
        
        # Train
        lstm_engine = ModelEngine()
        lstm_engine.train(train_df)
        
        # Predict on Test Set
        # Since predicts are sequential, we iterate through test set
        # But we need 10 days of context.
        # So for test_df[i], we need context from train_df + test_df[:i]
        full_df = pd.concat([train_df.tail(lstm_engine.sequence_length), test_df])
        
        lstm_preds = []
        lstm_true = test_df[target_col].values
        
        for i in range(len(test_df)):
            # Context window ending at current test point
            # seq_end_idx is aligned with test_df[i]
            # full_df index: i + sequence_length
            current_idx = i + lstm_engine.sequence_length
            seq_start = current_idx - lstm_engine.sequence_length + 1
            if seq_start < 0: seq_start = 0
            
            # Slice sequence
            seq_data = full_df.iloc[seq_start : current_idx + 1][feature_cols]
            
            # Predict
            probs_dict = lstm_engine.predict_probabilities(seq_data)
            
            # Convert probabilities back to class label
            # Mapping: 'y (↑)': 1, 'x (→)': 0, "x' (←)": 3, "y' (↓)": 2
            # We need to find max prob key and map to 0-3
            max_key = max(probs_dict, key=probs_dict.get)
            
            label_map = {
                "x (→)": 0.0,
                "y (↑)": 1.0, 
                "y' (↓)": 2.0,
                "x' (←)": 3.0
            }
            pred_label = label_map[max_key]
            lstm_preds.append(pred_label)
            
        acc_lstm = accuracy_score(lstm_true, lstm_preds)
        results.append({"Model": "LSTM (Proposed)", "Accuracy": acc_lstm})
        print(f"LSTM Accuracy: {acc_lstm:.2%}")

        # --- MODEL 2: Random Forest (Baseline ML) ---
        print("\n[3/5] Evaluating Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(train_df[feature_cols], train_df[target_col])
        rf_preds = rf.predict(test_df[feature_cols])
        acc_rf = accuracy_score(test_df[target_col], rf_preds)
        results.append({"Model": "Random Forest", "Accuracy": acc_rf})
        print(f"Random Forest Accuracy: {acc_rf:.2%}")

        # --- MODEL 3: SVM (Classic) ---
        print("\n[4/5] Evaluating SVM...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train_df[feature_cols])
        X_test_scaled = scaler.transform(test_df[feature_cols])
        
        svm = SVC(kernel='rbf', random_state=42)
        svm.fit(X_train_scaled, train_df[target_col])
        svm_preds = svm.predict(X_test_scaled)
        acc_svm = accuracy_score(test_df[target_col], svm_preds)
        results.append({"Model": "SVM", "Accuracy": acc_svm})
        print(f"SVM Accuracy: {acc_svm:.2%}")

        # --- MODEL 4: Dummy (Baseline) ---
        print("\n[5/5] Evaluating Dummy Classifier...")
        dummy = DummyClassifier(strategy='most_frequent')
        dummy.fit(train_df[feature_cols], train_df[target_col])
        dummy_preds = dummy.predict(test_df[feature_cols])
        acc_dummy = accuracy_score(test_df[target_col], dummy_preds)
        results.append({"Model": "Dummy (Baseline)", "Accuracy": acc_dummy})
        print(f"Dummy Accuracy: {acc_dummy:.2%}")

        # Report
        print("\n--- Final Results ---")
        res_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
        print(res_df)
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Accuracy", y="Model", data=res_df, hue="Model", palette="viridis")
        plt.title(f"Model Accuracy Comparison ({ticker})", fontsize=15)
        plt.xlabel("Accuracy Score", fontsize=12)
        plt.xlim(0, 1.0)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.savefig("model_comparison.png")
        print("Comparison chart saved to model_comparison.png")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_benchmark()
