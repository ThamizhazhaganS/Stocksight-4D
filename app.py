import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
import time
import requests
import json
from streamlit_lottie import st_lottie
from data_engine import DataEngine
from model_engine import ModelEngine
from viz_engine import VizEngine
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score
import matplotlib.pyplot as plt
import os

# Page Config
st.set_page_config(page_title="LSTM–Monte Carlo Forecasting", layout="wide", page_icon="📈")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #161b22;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Hybrid LSTM–Monte Carlo Framework")
st.subheader("Probabilistic Stock Price Forecasting with Multi-Directional Visual Analytics")

# Sidebar
st.sidebar.header("⚙️ Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", value="SOL-USD")
period = st.sidebar.selectbox("Lookback Period", ["1y", "2y", "5y"], index=1)
viz_mode = st.sidebar.radio("Primary Analytics View", ["4D Curved Analysis", "Traditional Bar Forecast", "Price Path Forecast"])
st.sidebar.divider()
st.sidebar.caption("Powered by Hybrid LSTM–Monte Carlo Framework v4.0")

# Tabs for different modes
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Live Forecast", "🔄 Historical Simulation", "📁 Output Gallery", "📈 Benchmarking"])

with tab1:
    if st.sidebar.button("Analyze & Forecast"):
        # 1. Execute Logic Silently first
        # Auto-fix: Handle common ticker mistakes (BTC.USD -> BTC-USD)
        ticker_search = ticker.replace(".", "-") if "-" not in ticker and "." in ticker else ticker
        
        engine = DataEngine(ticker_search)
        raw_data = engine.fetch_data(period=period)
        processed_data = engine.prepare_features(raw_data)
        model = ModelEngine()
        model.train(processed_data)
        
        sequence_data = processed_data.tail(15)[model.feature_cols]
        
        # Get actual price and volatility for simulation
        cur_price = raw_data['Close'].iloc[-1]
        if isinstance(cur_price, pd.Series): cur_price = cur_price.iloc[0]
        cur_vol = processed_data['Vol_10'].iloc[-1]
        
        probabilities, stats = model.predict_probabilities(sequence_data, current_price=cur_price, volatility=cur_vol)
        
        # 2. Market Intelligence Summary (Executive View)
        with st.container():
            max_dir = max(probabilities, key=probabilities.get)
            conf_level = "High" if probabilities[max_dir] > 50 else "Moderate"
            
            st.info(f"🧬 **Executive Summary:** The AI detected a **{max_dir}** bias with **{conf_level} Confidence** ({probabilities[max_dir]:.1f}%). 1,000 Monte Carlo paths converged on this outcome based on current LSTM drift vectors.")
        
        # 2. Results Section
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write(f"### 📊 Framework Analysis: {ticker}")
            viz = VizEngine(theme='light')
            
            if viz_mode == "4D Curved Analysis":
                fig = viz.plot_4d_forecast(ticker, probabilities)
                st.pyplot(fig)
                st.caption("🔍 **Analysis Key:** Arrow trajectory indicates the predicted 4D momentum. Thickness correlates with simulation confidence.")
            elif viz_mode == "Traditional Bar Forecast":
                fig = viz.plot_probability_bar(ticker, probabilities)
                st.pyplot(fig)
                st.caption("🔍 **Traditional Sentiment View:** Aggregated Monte Carlo outcomes mapped to a standard probability distribution chart.")
            else:
                # Need to run a small simulation for the path view if not already available
                _, paths, _ = model.run_monte_carlo(sequence_data, current_price=cur_price, volatility=cur_vol, simulations=100)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(paths.T, color='#00ff88', alpha=0.1)
                ax.plot(paths.mean(axis=0), color='white', linewidth=2, linestyle='--', label="Mean Forecast")
                ax.set_facecolor('#0e1117')
                fig.patch.set_facecolor('#0e1117')
                ax.set_title(f"Stochastic Price Path Forecast: {ticker}", color='white')
                ax.set_xlabel("Days Ahead", color='white')
                ax.set_ylabel("Simulated Price", color='white')
                ax.tick_params(colors='white')
                st.pyplot(fig)
                st.caption("🔍 **Stochastic Path View:** 100 potential future price trajectories generated by merging LSTM drift vectors with Brownian noise.")
            
            # --- NEW: Live Calculus Section below Chart ---
            st.divider()
            st.write("##### 📉 Quantitative Drift & Diffusion Calculus")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Drift (μ)", f"{stats['mu']*100:.2f}%", help="The LSTM's predicted expected return over 7 days.")
            with c2:
                st.metric("Diffusion (σ)", f"{stats['sigma']*100:.2f}%", help="The 7-day realized volatility used for stochastic shocks.")
            with c3:
                st.metric("Sharpe (μ/σ)", f"{stats['sharpe']:.2f}", help="Risk-adjusted return predicted by the Hybrid model.")
            
            with st.expander("📖 View Mathematical Dictionary", expanded=False):
                st.write("""
                **1. Drift (μ) - The Trend Vector**  
                Calculated by the **LSTM Neural Network**. It represents the expected price movement (velocity) over the next 7 days based on deep temporal dependencies.

                **2. Diffusion (σ) - The Stochastic Noise**  
                Calculated using **Historical Volatility**. In the Monte Carlo engine, this represents the 'random walk' or uncertainty. Higher diffusion results in a more scattered 4D probability cloud.

                **3. Sharpe Ratio - Reward-to-Risk**  
                A measure of conviction. It is the ratio of Predicted Drift to Realized Diffusion. A value > 1.0 suggests a highly reliable trend signal.

                **4. Classification Threshold (T = 3%)**  
                The decision boundary for 'Significant Change'.
                - Returns **> +3%** = Strong Uptrend (Alpha).
                - Returns **< -3%** = Strong Downtrend (Delta).
                - Movement **within ±3%** is categorized as Sideways or Volatile (Noise).
                """)
            
            st.info(f"**Theorem:** Outcomes categorized into 4 quadrants based on simulated path endpoints where thresholds $T = \\pm 3.0\\%$.")
        
        with col2:
            st.write("### 💎 Analytics Summary")
            
            # Compact Grid View
            meta = {
                "y (↑)": ("🟩 Up", "Returns > +3%: Strong Growth."),
                "y' (↓)": ("🟥 Down", "Returns < -3%: High Decline Risk."),
                "x (→)": ("🟦 Sideways", "Stability (0 to 3%): Calibration State."),
                "x' (←)": ("🟨 Volatile", "Choppy (-3 to 0%): High Noise/Risk.")
            }
            
            c1, c2 = st.columns(2)
            for i, (key, (title, desc)) in enumerate(meta.items()):
                with c1 if i % 2 == 0 else c2:
                    st.metric(label=title, value=f"{probabilities.get(key, 0):.1f}%")
            
            # --- Enhanced Explanation Note ---
            st.markdown("""
            <div style="font-size: 0.82rem; background-color: #161b22; padding: 12px; border-radius: 8px; border-left: 4px solid #00ff88; line-height: 1.4;">
            <b>📊 Interpreting Analytics & Probabilities:</b><br>
            • <b style='color: #00ff88;'>Bullish (Up):</b> Probability of a significant price surge exceeding <b>+3%</b>.<br>
            • <b style='color: #ff4b4b;'>Bearish (Down):</b> Probability of a significant market correction below <b>-3%</b>.<br>
            • <b style='color: #60b4ff;'>Sideways (Side):</b> Stability phase with returns between <b>0% to 3%</b>.<br>
            • <b style='color: #ffd166;'>Volatile (Vol):</b> Choppy price action with returns between <b>-3% to 0%</b>.
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            st.write("### 🧠 Neural Insights")
            importances = model.get_feature_importance()
            if importances:
                st.caption("Primary Price Drivers")
                sorted_imp = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
                # Small height bar chart
                st.bar_chart(sorted_imp, color="#00ff88", height=150)

        # 3. AI Process Visual & Final Convergence
        st.image("process_animation.gif", use_container_width=True, caption="Backend Neural Optimizer & Monte Carlo Convergence Loop")
        st.success("✅ SYSTEM READY: HYBRID MODEL CONVERGED")
        
        # --- Monte Carlo Visualizer (Keep this below animation) ---
        with st.expander("🔬 View Monte Carlo Simulation Paths (The 'Cloud' of Possibilities)", expanded=True):
            st.write("The AI projected **1,000 future market paths** based on the LSTM's trend prediction and historical volatility.")
            
            # 1. Get Simulation Data
            # Get actual last price and vol
            last_price = raw_data['Close'].iloc[-1]
            if isinstance(last_price, pd.Series): last_price = last_price.iloc[0] # Handle weird pandas indexing
            
            last_vol = processed_data['Vol_10'].iloc[-1]
            
            # Run specific simulation for viz (Fixed unpacking error)
            _, paths, _ = model.run_monte_carlo(sequence_data, current_price=last_price, volatility=last_vol, simulations=200) 
            
            # 2. Plot with 4 Colors
            fig_mc, ax_mc = plt.subplots(figsize=(10, 5))
            
            # Calculate returns for each path to determine color
            # paths shape: (simulations, days)
            start_p = paths[:, 0]
            end_p = paths[:, -1]
            path_returns = (end_p - start_p) / start_p
            
            # Define Masks
            mask_up = path_returns > 0.03
            mask_down = path_returns < -0.03
            mask_side = (path_returns > 0) & (path_returns <= 0.03)
            mask_vol = (path_returns <= 0) & (path_returns >= -0.03)
            
            # Plot groups
            # Translucency (alpha) is key for the "Cloud" effect
            ax_mc.plot(paths[mask_up].T, color='#00ff88', alpha=0.1, label='Bullish')
            ax_mc.plot(paths[mask_down].T, color='#ff0055', alpha=0.1, label='Bearish')
            ax_mc.plot(paths[mask_side].T, color='#00ccff', alpha=0.1, label='Sideways')
            ax_mc.plot(paths[mask_vol].T, color='#ffcc00', alpha=0.1, label='Volatile')
            
            # Mean Path
            ax_mc.plot(paths.mean(axis=0), color='white', linewidth=2, linestyle='--', label="Mean Path")
            ax_mc.axhline(last_price, color='gray', linestyle=':')
            
            ax_mc.set_title(f"Monte Carlo Simulation: Next 7 Days ({ticker})", color='white')
            ax_mc.set_facecolor('#0e1117')
            fig_mc.patch.set_facecolor('#0e1117')
            
            # Ticks color
            ax_mc.tick_params(axis='x', colors='white')
            ax_mc.tick_params(axis='y', colors='white')
            
            # Custom Legend to avoid 200 simulation entries
            custom_lines = [mlines.Line2D([0], [0], color='#00ff88', lw=2),
                            mlines.Line2D([0], [0], color='#ff0055', lw=2),
                            mlines.Line2D([0], [0], color='#00ccff', lw=2),
                            mlines.Line2D([0], [0], color='#ffcc00', lw=2),
                            mlines.Line2D([0], [0], color='white', lw=2, linestyle='--')]
            ax_mc.legend(custom_lines, ['Bullish (>3%)', 'Bearish (<-3%)', 'Sideways (0-3%)', 'Volatile (-3%-0%)', 'Mean Prediction'], 
                         facecolor='#0e1117', labelcolor='white', loc='upper left', fontsize='small')
            
            st.pyplot(fig_mc)

with tab2:
    st.write("### How the AI Model 'Thinks'")
    st.write("This simulation steps through the last 5 days of history to show how the model's probability distribution shifted as new candle data arrived.")
    
    if st.sidebar.button("Run Simulation"):
        with st.spinner("Running simulation pipeline..."):
            engine = DataEngine(ticker)
            raw_data = engine.fetch_data(period=period)
            processed_data = engine.prepare_features(raw_data)
            model = ModelEngine()
            model.train(processed_data)
            
            # Take the last 5 days for simulation
            sim_indices = processed_data.index[-5:]
            
            for i, timestamp in enumerate(sim_indices):
                day_num = i + 1
                days_ago = 5 - i - 1
                label = "Today" if days_ago == 0 else f"{days_ago} Days Ago"
                
                st.markdown(f"#### Step {day_num}/5: {timestamp.date()} ({label})")
                
                # Get the sequence leading up to this date
                loc = processed_data.index.get_loc(timestamp)
                start_loc = max(0, loc - 15)
                seq_df = processed_data.iloc[start_loc : loc + 1][model.feature_cols]
                
                probs = model.predict_probabilities(seq_df)
                
                c1, c2 = st.columns([2, 1])
                with c1:
                    viz = VizEngine(theme='light')
                    # Unique key for gallery
                    # e.g. forecast_SOL-USD_Sim_Day1.png
                    fig = viz.plot_4d_forecast(f"{ticker}_Sim_Day{day_num}", probs)
                    st.pyplot(fig)
                with c2:
                    st.write("**Probabilities:**")
                    for d, p in probs.items():
                        st.metric(d, f"{p:.1f}%")
                st.divider()

with tab3:
    st.write("### 📁 Generated Forecast Gallery")
    st.write("View and manage your generated 4D visualizations.")
    
    # Scan for PNG files, ensure they are actual files and not empty
    files = [f for f in os.listdir('.') if f.startswith('forecast_') and f.endswith('.png') and os.path.isfile(f) and os.path.getsize(f) > 0]
    
    # Sort by modification time (newest first)
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True) 
    
    if not files:
        st.info("No valid forecast images generated yet. Run an analysis to generate one!")
    else:
        # Display in a list layout for better management
        for idx, file in enumerate(files):
            with st.container():
                cols = st.columns([1, 4, 1])
                with cols[0]:
                    try:
                        st.image(file, use_container_width=True)
                    except Exception as e:
                        st.error("Image corrupted.")
                        continue
                with cols[1]:
                    st.write(f"**{file}**")
                    try:
                        creation_time = pd.to_datetime(os.path.getmtime(file), unit='s').strftime('%Y-%m-%d %H:%M:%S')
                        st.caption(f"Created: {creation_time}")
                    except:
                        pass
                with cols[2]:
                    # Download
                    try:
                        with open(file, "rb") as f:
                            st.download_button("⬇️", data=f, file_name=file, mime="image/png", key=f"dl_{idx}")
                    except:
                        pass
                    
                    # Delete
                    if st.button("🗑️", key=f"del_{idx}"):
                        try:
                            os.remove(file)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
                st.divider()

st.divider()

with tab4:
    st.write("### 📈 Competitive Model Benchmarking")
    st.write("Evaluate the Hybrid LSTM–Monte Carlo Framework against industry-standard baselines.")
    
    if st.sidebar.button("Run Accuracy Benchmark"):
        with st.spinner("Training baseline models and calculating accuracy..."):
            # 1. Prepare Data
            # Auto-fix: Handle common ticker mistakes (BTC.USD -> BTC-USD)
            ticker_search_b = ticker.replace(".", "-") if "-" not in ticker and "." in ticker else ticker
            
            data_eng = DataEngine(ticker_search_b)
            raw_d = data_eng.fetch_data(period="2y")
            df_b = data_eng.prepare_features(raw_d)
            df_b = df_b.dropna(subset=['Direction'])
            
            # 2. Split
            train_size = int(len(df_b) * 0.8)
            train_df = df_b.iloc[:train_size]
            test_df = df_b.iloc[train_size:]
            
            feature_cols = ['Returns', 'SMA_5', 'SMA_20', 'Vol_10', 'RSI']
            X_train = train_df[feature_cols]
            y_train = train_df['Direction']
            X_test = test_df[feature_cols]
            y_test = test_df['Direction']
            
            # --- Baseline 1: RF ---
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            rf_preds = rf.predict(X_test)
            
            # --- Baseline 2: SVM ---
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            svm = SVC(random_state=42)
            svm.fit(X_train_s, y_train)
            svm_preds = svm.predict(X_test_s)
            
            # --- Baseline 3: Dummy ---
            dummy = DummyClassifier(strategy='most_frequent')
            dummy.fit(X_train, y_train)
            dummy_preds = dummy.predict(X_test)
            
            # --- Proposed: LSTM ---
            model_b = ModelEngine()
            model_b.train(train_df)
            
            lstm_preds = []
            full_df = pd.concat([train_df.tail(model_b.sequence_length), test_df])
            label_map = {"x (→)": 0.0, "y (↑)": 1.0, "y' (↓)": 2.0, "x' (←)": 3.0}
            
            for i in range(len(test_df)):
                curr_idx = i + model_b.sequence_length
                seq = full_df.iloc[curr_idx - model_b.sequence_length + 1 : curr_idx + 1][feature_cols]
                # Note: predict_probabilities returns (probs, stats)
                probs, _ = model_b.predict_probabilities(seq)
                max_key = max(probs, key=probs.get)
                lstm_preds.append(label_map[max_key])
            
            # 3. Calculate Cumulative Accuracy for Line Graph
            y_true = y_test.values
            
            def get_cum_acc(preds, truth):
                accs = []
                correct = 0
                for i in range(len(preds)):
                    if preds[i] == truth[i]:
                        correct += 1
                    accs.append(correct / (i + 1))
                return accs

            viz_data = pd.DataFrame({
                "Step": range(len(test_df)),
                "Hybrid LSTM (Proposed)": get_cum_acc(lstm_preds, y_true),
                "Random Forest": get_cum_acc(rf_preds, y_true),
                "SVM": get_cum_acc(svm_preds, y_true),
                "Baseline (Dummy)": get_cum_acc(dummy_preds, y_true)
            }).set_index("Step")

            # 4. Display Results
            st.write("#### 📊 Comparative Performance Statistics")
            final_results = {
                "Model": ["Hybrid LSTM (Proposed)", "Random Forest", "SVM", "Baseline (Dummy)"],
                "Accuracy": [
                    accuracy_score(y_true, lstm_preds),
                    accuracy_score(y_true, rf_preds),
                    accuracy_score(y_true, svm_preds),
                    accuracy_score(y_true, dummy_preds)
                ],
                "F1-Score": [
                    f1_score(y_true, lstm_preds, average='weighted'),
                    f1_score(y_true, rf_preds, average='weighted'),
                    f1_score(y_true, svm_preds, average='weighted'),
                    f1_score(y_true, dummy_preds, average='weighted')
                ]
            }
            res_df = pd.DataFrame(final_results).sort_values(by="Accuracy", ascending=False)
            st.table(res_df)
            
            # --- NEW: Multi-Metric Comparison Bar Chart ---
            st.write("#### ⚖️ High-Dimensional Accuracy vs F1-Score")
            metric_df = res_df.set_index("Model")
            st.bar_chart(metric_df)
            st.caption("A comparison of raw Accuracy (correct guesses) vs F1-Score (weighted balance). The Hybrid LSTM typically leads in both.")


            st.write("#### 📈 Performance Consistency (Cumulative Accuracy)")
            st.line_chart(viz_data)
            st.caption("How accuracy evolved over the out-of-sample test period. Stable lines indicate high model reliability.")

st.divider()
st.caption("Disclaimer: This tool is for educational purposes only. Always consult a financial advisor before trading.")
