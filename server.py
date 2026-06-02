from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from fastapi.responses import FileResponse
import os

from data_engine import DataEngine
from model_engine import ModelEngine
from viz_engine import VizEngine

app = FastAPI(title="Stocksight-4D API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    ticker: str
    period: str = "2y"

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

@app.get("/api/process_animation")
def get_animation():
    if os.path.exists("process_animation.gif"):
        return FileResponse("process_animation.gif")
    raise HTTPException(status_code=404, detail="Animation not found")

import asyncio

@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    ticker = req.ticker.replace(".", "-") if "-" not in req.ticker and "." in req.ticker else req.ticker
    period = req.period

    try:
        engine = DataEngine(ticker)
        raw_data = await asyncio.to_thread(engine.fetch_data, period=period)
        processed_data = engine.prepare_features(raw_data)
        model = ModelEngine()
        await asyncio.to_thread(model.train, processed_data)

        sequence_data = processed_data.tail(15)[model.feature_cols]

        cur_price = raw_data['Close'].iloc[-1]
        if isinstance(cur_price, pd.Series): cur_price = cur_price.iloc[0]
        cur_vol = processed_data['Vol_10'].iloc[-1]

        # Use 100 simulations for speed in web requests, or maybe 500
        probs, stats = model.predict_probabilities(sequence_data, current_price=cur_price, volatility=cur_vol)

        max_dir = max(probs, key=probs.get)
        conf_level = "High" if probs[max_dir] > 50 else "Moderate"
        exec_summary = f"The AI detected a {max_dir} bias with {conf_level} Confidence ({probs[max_dir]:.1f}%). 1,000 Monte Carlo paths converged on this outcome based on current LSTM drift vectors."

        viz = VizEngine(theme='dark')

        # 1. 4D Curved Analysis
        fig_4d = viz.plot_4d_forecast(ticker, probs)
        b64_4d = fig_to_base64(fig_4d)

        # 2. Probability Bar
        fig_bar = viz.plot_probability_bar(ticker, probs)
        b64_bar = fig_to_base64(fig_bar)

        # 3. Path Forecast (100 paths)
        _, paths_100, _ = model.run_monte_carlo(sequence_data, current_price=cur_price, volatility=cur_vol, simulations=100)
        fig_path, ax = plt.subplots(figsize=(10, 5))
        ax.plot(paths_100.T, color='#00ff88', alpha=0.1)
        ax.plot(paths_100.mean(axis=0), color='white', linewidth=2, linestyle='--', label="Mean Forecast")
        ax.set_facecolor('#0e1117')
        fig_path.patch.set_facecolor('#0e1117')
        ax.set_title(f"Stochastic Price Path Forecast: {ticker}", color='white')
        ax.set_xlabel("Days Ahead", color='white')
        ax.set_ylabel("Simulated Price", color='white')
        ax.tick_params(colors='white')
        b64_path = fig_to_base64(fig_path)

        # 4. Monte Carlo colored
        _, paths_mc, _ = model.run_monte_carlo(sequence_data, current_price=cur_price, volatility=cur_vol, simulations=200)
        fig_mc, ax_mc = plt.subplots(figsize=(10, 5))
        start_p = paths_mc[:, 0]
        end_p = paths_mc[:, -1]
        path_returns = (end_p - start_p) / start_p

        mask_up = path_returns > 0.03
        mask_down = path_returns < -0.03
        mask_side = (path_returns > 0) & (path_returns <= 0.03)
        mask_vol = (path_returns <= 0) & (path_returns >= -0.03)

        if mask_up.any(): ax_mc.plot(paths_mc[mask_up].T, color='#00ff88', alpha=0.1, label='Bullish')
        if mask_down.any(): ax_mc.plot(paths_mc[mask_down].T, color='#ff0055', alpha=0.1, label='Bearish')
        if mask_side.any(): ax_mc.plot(paths_mc[mask_side].T, color='#00ccff', alpha=0.1, label='Sideways')
        if mask_vol.any(): ax_mc.plot(paths_mc[mask_vol].T, color='#ffcc00', alpha=0.1, label='Volatile')

        ax_mc.plot(paths_mc.mean(axis=0), color='white', linewidth=2, linestyle='--', label="Mean Path")
        ax_mc.axhline(cur_price, color='gray', linestyle=':')

        ax_mc.set_title(f"Monte Carlo Simulation: Next 7 Days ({ticker})", color='white')
        ax_mc.set_facecolor('#0e1117')
        fig_mc.patch.set_facecolor('#0e1117')
        ax_mc.tick_params(axis='x', colors='white')
        ax_mc.tick_params(axis='y', colors='white')

        custom_lines = [
            mlines.Line2D([0], [0], color='#00ff88', lw=2),
            mlines.Line2D([0], [0], color='#ff0055', lw=2),
            mlines.Line2D([0], [0], color='#00ccff', lw=2),
            mlines.Line2D([0], [0], color='#ffcc00', lw=2),
            mlines.Line2D([0], [0], color='white', lw=2, linestyle='--')
        ]
        ax_mc.legend(custom_lines, ['Bullish (>3%)', 'Bearish (<-3%)', 'Sideways (0-3%)', 'Volatile (-3%-0%)', 'Mean Prediction'],
                     facecolor='#0e1117', labelcolor='white', loc='upper left', fontsize='small')

        b64_mc = fig_to_base64(fig_mc)

        return {
            "ticker": ticker,
            "executive_summary": exec_summary,
            "stats": stats,
            "probabilities": probs,
            "feature_importances": model.get_feature_importance(),
            "charts": {
                "four_d": f"data:image/png;base64,{b64_4d}",
                "bar": f"data:image/png;base64,{b64_bar}",
                "path": f"data:image/png;base64,{b64_path}",
                "monte_carlo": f"data:image/png;base64,{b64_mc}"
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/gallery")
def get_gallery():
    files = [f for f in os.listdir('.') if f.startswith('forecast_') and f.endswith('.png') and os.path.isfile(f) and os.path.getsize(f) > 0]
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    gallery = []
    for f in files:
        creation_time = pd.to_datetime(os.path.getmtime(f), unit='s').strftime('%Y-%m-%d %H:%M:%S')
        with open(f, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            gallery.append({"filename": f, "b64": f"data:image/png;base64,{encoded_string}", "created_at": creation_time})
    return {"images": gallery}

@app.delete("/api/gallery/{filename}")
def delete_gallery_image(filename: str):
    if filename.startswith('forecast_') and filename.endswith('.png') and os.path.exists(filename):
        os.remove(filename)
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="File not found or access denied")

@app.post("/api/simulate")
def simulate(req: AnalyzeRequest):
    ticker = req.ticker.replace(".", "-") if "-" not in req.ticker and "." in req.ticker else req.ticker
    try:
        engine = DataEngine(ticker)
        raw_data = engine.fetch_data(period=req.period)
        processed_data = engine.prepare_features(raw_data)
        model = ModelEngine()
        model.train(processed_data)
        
        sim_indices = processed_data.index[-5:]
        results = []
        viz = VizEngine(theme='dark')
        
        for i, timestamp in enumerate(sim_indices):
            day_num = i + 1
            days_ago = 5 - i - 1
            label = "Today" if days_ago == 0 else f"{days_ago} Days Ago"
            
            loc = processed_data.index.get_loc(timestamp)
            start_loc = max(0, loc - 15)
            seq_df = processed_data.iloc[start_loc : loc + 1][model.feature_cols]
            
            probs, _ = model.predict_probabilities(seq_df)
            fig_4d = viz.plot_4d_forecast(f"{ticker}_Sim_Day{day_num}", probs)
            b64_4d = fig_to_base64(fig_4d)
            
            results.append({
                "day_num": day_num,
                "date": timestamp.date().isoformat(),
                "label": label,
                "probabilities": probs,
                "chart": f"data:image/png;base64,{b64_4d}"
            })
            
        return {"steps": results}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/benchmark")
async def run_benchmark(req: AnalyzeRequest):
    ticker = req.ticker.replace(".", "-") if "-" not in req.ticker and "." in req.ticker else req.ticker
    try:
        from sklearn.metrics import accuracy_score
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.dummy import DummyClassifier
        from sklearn.preprocessing import StandardScaler
        
        engine = DataEngine(ticker)
        raw_data = await asyncio.to_thread(engine.fetch_data, period=req.period)
        df = engine.prepare_features(raw_data)
        df = df.dropna(subset=['Direction'])
        
        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size].copy()
        test_df = df.iloc[train_size:].copy()
        
        feature_cols = ['Returns', 'SMA_5', 'SMA_20', 'Vol_10', 'RSI']
        target_col = 'Direction'
        
        lstm_engine = ModelEngine()
        await asyncio.to_thread(lstm_engine.train, train_df)
        
        # Test on last 50 days (or fewer if test_df is small)
        test_samples = min(50, len(test_df))
        test_subset = test_df.tail(test_samples)
        
        lstm_preds = []
        for i in range(len(test_subset)):
            current_idx = i + len(df) - test_samples
            seq_start = current_idx - lstm_engine.sequence_length + 1
            if seq_start < 0: seq_start = 0
            seq_data = df.iloc[seq_start : current_idx + 1][feature_cols]
            probs, _ = lstm_engine.predict_probabilities(seq_data, current_price=100.0, volatility=0.2)
            max_key = max(probs, key=probs.get)
            label_map = {"x (→)": 0.0, "y (↑)": 1.0, "y' (↓)": 2.0, "x' (←)": 3.0}
            lstm_preds.append(label_map[max_key])
            
        lstm_acc = accuracy_score(test_subset[target_col].values, lstm_preds)
        
        # Random Forest
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(train_df[feature_cols], train_df[target_col])
        rf_acc = accuracy_score(test_subset[target_col], rf.predict(test_subset[feature_cols]))
        
        # SVM
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train_df[feature_cols])
        svm = SVC(kernel='rbf', random_state=42)
        svm.fit(X_train_scaled, train_df[target_col])
        svm_acc = accuracy_score(test_subset[target_col], svm.predict(scaler.transform(test_subset[feature_cols])))
        
        # Dummy
        dummy = DummyClassifier(strategy='most_frequent')
        dummy.fit(train_df[feature_cols], train_df[target_col])
        dummy_acc = accuracy_score(test_subset[target_col], dummy.predict(test_subset[feature_cols]))
        
        results = [
            {"Model": "Hybrid LSTM (Proposed)", "Accuracy": round(float(lstm_acc) * 100, 2)},
            {"Model": "Random Forest", "Accuracy": round(float(rf_acc) * 100, 2)},
            {"Model": "SVM", "Accuracy": round(float(svm_acc) * 100, 2)},
            {"Model": "Baseline Dummy", "Accuracy": round(float(dummy_acc) * 100, 2)}
        ]
        
        # Sort
        results.sort(key=lambda x: x["Accuracy"], reverse=True)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        models = [r["Model"] for r in results]
        accs = [r["Accuracy"] for r in results]
        
        colors = ['#00ff88' if 'LSTM' in m else '#3b82f6' for m in models]
        bars = ax.barh(models, accs, color=colors)
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        ax.set_xlabel("Accuracy (%)", color='white')
        ax.set_title(f"Model Accuracy Comparison: {ticker} (Last {test_samples} Days)", color='white')
        ax.invert_yaxis()
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center', color='white', fontweight='bold')
            
        b64_chart = fig_to_base64(fig)
        
        return {
            "ticker": ticker,
            "results": results,
            "chart": f"data:image/png;base64,{b64_chart}"
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Mount the static frontend directory
if os.path.exists("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")
