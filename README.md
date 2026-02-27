# Hybrid LSTM–Monte Carlo Framework 🛡️

**Probabilistic Stock Price Forecasting with Multi-Directional Visual Analytics**

This project is a final-year BE CSE AIML project that implements a state-of-the-art framework for quantifying market uncertainty. It combines Deep Learning (LSTM) for trend prediction with Monte Carlo simulations to generate robust probability distributions in four key directions:
- **Up (↑)**: Significant Bullish Growth
- **Down (↓)**: Significant Bearish Decline
- **Right (→)**: Sideways / Consolidation
- **Left (←)**: High Volatility / Reversal

## 🚀 Key Features
- **Dynamic Data Fetching:** Uses Yahoo Finance API to get live historical market data.
- **Hybrid AI Engine:** PyTorch LSTM Regressor for drift estimation + Monte Carlo Simulation for probabilistic path analysis.
- **Multi-Directional Visual Analytics:** Custom 4D visualization system to visualize market bias and risk zones.
- **Real-time Pipeline Animation:** Visualizes the internal AI processing steps during inference.

## 🛠️ Tech Stack
- **Python** (Core)
- **yfinance** (Data Source)
- **PyTorch** (LSTM Neural Network)
- **Matplotlib** (Scientific Visualization)
- **Streamlit** (Interactive Dashboard)

## 📦 Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🖥️ Usage

### 1. Interactive Dashboard (Recommended)
Run the Streamlit app for a premium UI experience:
```bash
streamlit run app.py
```

### 2. Command Line Interface (CLI)
Run the analysis for a specific ticker (e.g., Apple):
```bash
python main.py AAPL
```
This will generate a `forecast_AAPL.png` image with the results.

## 📂 Project Structure
- `app.py`: Streamlit dashboard.
- `main.py`: CLI entry point.
- `data_engine.py`: Handles data fetching and indicator calculation.
- `model_engine.py`: AI training and probability prediction.
- `viz_engine.py`: Custom 4D curved arrow plotting logic.
- `requirements.txt`: Python dependencies.

---
**Disclaimer:** This tool is for educational purposes as part of a BE CSE AIML final year project. It provides decision support analysis and is not financial advice.
