# Hybrid LSTM–Monte Carlo Framework for Probabilistic Stock Price Forecasting with Multi-Directional Visual Analytics

## 1. System Architecture Diagram

The system follows a **Hybrid Deep Learning + Monte Carlo Simulation** architecture:

```mermaid
graph TD
    A[User Input (Ticker)] --> B[Data Engine]
    B -->|Fetch Live Data| C((Yahoo Finance API))
    C -->|Raw OHLCV Data| B
    B -->|Compute Indicators| D[Feature Engineering]
    D -->|Sliding Window| E[Sequence Creation]
    E -->|Sequence Tensor| F[LSTM Regressor (PyTorch)]
    F -->|Expected 7D Return| G[Monte Carlo Engine]
    G -->|1,000 Random Paths| H[Probability Distribution]
    H -->|Quadrant Classification| I[Visualization Engine]
    I -->|4D Process Visual| J[Streamlit Dashboard]

    subgraph "Hybrid AI Core"
    F
    G
    H
    end

    subgraph "Presentation"
    I
    J
    end
```

### **Data Flow:**
1.  **Input:** User selects a stock ticker (e.g., "SOL-USD").
2.  **Acquisition:** `data_engine.py` fetches the last 2 years of daily data.
3.  **Processing:** Technical indicators (RSI, SMA, Volatility) are engineered.
4.  **Regression:** The **LSTM** predicts the *continuous* future return (Expected Drift).
5.  **Simulation:** The **Monte Carlo Engine** runs 1,000 simulations using Geometric Brownian Motion (GBM) around that predicted drift.
6.  **Probabilities:** The final endpoint of the 1,000 paths are categorized into Up, Down, Sideways, or Volatile based on specific return thresholds.

---

## 2. Mathematical Calculations

### **A. Geometric Brownian Motion (GBM) for Monte Carlo**
The simulation is driven by the stochastic differential equation:
$$ dS_t = \mu S_t dt + \sigma S_t dW_t $$
Where:
- $\mu$: Predicted drift from LSTM.
- $\sigma$: Historical 10-day volatility.
- $dW_t$: Random Wiener process (Normal distribution shocks).

The AI calculates 1,000 possible future price paths for the next 7 days.

### **B. Classification of Simulated Outcomes**
After 1,000 simulations, the model calculates probabilities:
- **Bullish (↑)**: Final Return > +3%
- **Bearish (↓)**: Final Return < -3%
- **Sideways (→)**: Final Return between 0% and +3%
- **Volatile (←)**: Final Return between -3% and 0%

---

## 3. Visualization Core

### **A. The 4D Forecast Chart**
- **North (y):** High Positive Momentum.
- **South (y'):** High Negative Momentum.
- **East (x):** Low Volatility Stability.
- **West (x'):** High Noise / Choppy Movement.
The arrow thickness is proportional to the probability confidence.

### **B. Process Animation (GIF)**
The dashboard includes an **AI Processing Visual** (generated via `create_animation.py`) that demonstrates the real-time optimization of the Neural Network as it executes the pipeline:
1.  **Data Ingestion Layer**
2.  **Hidden LSTM Layer Processing**
3.  **Monte Carlo Convergence**

---

## 4. Key Technologies
- **PyTorch:** Deep Learning framework for LSTM.
- **YFinance:** Real-time financial data bridge.
- **Matplotlib:** Advanced scientific plotting for the 4D charts and process animations.
- **Streamlit:** Modern web application framework.
