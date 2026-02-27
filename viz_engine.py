import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from datetime import datetime

class VizEngine:
    def __init__(self, theme='light'):
        self.theme = theme

    def draw_arrow_curved(self, ax, theta_deg, length, label, offset, width, curve_strength=0.8):
        theta_rad = np.radians(theta_deg)
        x_end = length * np.cos(theta_rad)
        y_end = length * np.sin(theta_rad)
        mid_x = x_end / 2
        mid_y = y_end / 2
        curve_dir_x = -np.sin(theta_rad)
        curve_dir_y = np.cos(theta_rad)
        control_x = mid_x + curve_strength * curve_dir_x * length * 0.3
        control_y = mid_y + curve_strength * curve_dir_y * length * 0.3
        verts = [(0, 0), (control_x, control_y), (x_end, y_end)]
        codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
        path = Path(verts, codes)
        
        # Use a minimum width for visibility
        line_width = max(1.0, width)
        patch = PathPatch(path, facecolor='none', edgecolor='black', lw=line_width)
        ax.add_patch(patch)
        
        arrow_size = 0.12
        angle = np.arctan2(y_end - control_y, x_end - control_x)
        for sign in [-1, 1]:
            x_side = x_end - arrow_size * np.cos(angle + sign * np.pi/6)
            y_side = y_end - arrow_size * np.sin(angle + sign * np.pi/6)
            ax.plot([x_end, x_side], [y_end, y_side], color='black', lw=line_width)
        
        ax.text(x_end + offset[0], y_end + offset[1], f"{label:.0f}%", fontsize=12, ha='center', va='center', fontweight='bold')

    def plot_4d_forecast(self, ticker, probabilities):
        """
        Creates the Hybrid LSTM–Monte Carlo 4D visualization.
        probabilities: dict with keys ['y (↑)', 'x (→)', "x' (←)", "y' (↓)"]
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.axis('off')

        # ✅ Fill Quadrants Based on Risk Color
        # Q2 (Top-Right in code mapping green)
        ax.axhspan(0, 3, xmin=0.5, xmax=1, color='green', alpha=0.3)      
        # Q4 (Bottom-Right in code mapping red)
        ax.axhspan(-3, 0, xmin=0.5, xmax=1, color='red', alpha=0.3)       
        # Q3 (Bottom-Left in code mapping yellow)
        ax.axhspan(-3, 0, xmin=0, xmax=0.5, color='yellow', alpha=0.3)    
        # Q1 (Top-Left in code mapping lightblue)
        ax.axhspan(0, 3, xmin=0, xmax=0.5, color='lightblue', alpha=0.3)  

        # Coordinate arrows
        ax.arrow(0, 0, 2.4, 0, head_width=0.1, head_length=0.1, color='black')
        ax.arrow(0, 0, -2.4, 0, head_width=0.1, head_length=0.1, color='black')
        ax.arrow(0, 0, 0, 2.4, head_width=0.1, head_length=0.1, color='black')
        ax.arrow(0, 0, 0, -2.4, head_width=0.1, head_length=0.1, color='black')
        ax.text(2.4, 0.1, "x", fontsize=12, ha='right')
        ax.text(-2.4, 0.1, "x'", fontsize=12, ha='left')
        ax.text(0.1, 2.4, "y", fontsize=12, va='top')
        ax.text(0.1, -2.4, "y'", fontsize=12, va='bottom')

        # Curved arrows based on user-provided angles
        # Right: 45 deg, Left: 225 deg, Up: 135 deg, Down: 315 deg
        self.draw_arrow_curved(ax, 45, 2, probabilities.get("x (→)", 0), (0.1, 0.1), probabilities.get("x (→)", 0)/10)
        self.draw_arrow_curved(ax, 225, 2, probabilities.get("x' (←)", 0), (-0.1, 0.1), probabilities.get("x' (←)", 0)/10)
        self.draw_arrow_curved(ax, 135, 2, probabilities.get("y (↑)", 0), (-0.1, 0.1), probabilities.get("y (↑)", 0)/10)
        self.draw_arrow_curved(ax, 315, 2, probabilities.get("y' (↓)", 0), (-0.1, -0.1), probabilities.get("y' (↓)", 0)/10)

        # Title with date and ticker
        current_date = datetime.now().strftime("%Y-%m-%d")
        ax.set_title(f"Hybrid LSTM–Monte Carlo Directional Forecast: {ticker}\n({current_date})", fontsize=14, pad=20, ha='center')
        ax.text(0, -3.2, "Next 7 Days Forecast", ha='center', fontsize=11, fontweight='bold')

        plt.tight_layout()
        
        # Save output
        output_path = f"forecast_{ticker}.png"
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
        return fig

    def plot_probability_bar(self, ticker, probabilities):
        """
        Creates a professional bar chart of probabilities.
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Mapping labels to colors
        data = {
            "Bullish (↑)": probabilities.get("y (↑)", 0),
            "Bearish (↓)": probabilities.get("y' (↓)", 0),
            "Sideways (→)": probabilities.get("x (→)", 0),
            "Volatile (←)": probabilities.get("x' (←)", 0)
        }
        
        colors = ['#00ff88', '#ff4b4b', '#60b4ff', '#ffd166'] # Green, Red, Blue, Yellow
        
        bars = ax.bar(data.keys(), data.values(), color=colors, edgecolor='white', linewidth=1.2)
        
        # Add values on top
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', 
                    color='white', fontweight='bold', fontsize=12)

        ax.set_ylim(0, max(data.values()) + 15)
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        
        ax.set_title(f"Market Sentiment Distribution: {ticker}", color='white', fontsize=14, pad=20)
        ax.set_ylabel("Probability (%)", color='white')
        
        ax.tick_params(axis='x', colors='white', labelsize=11)
        ax.tick_params(axis='y', colors='white')
        
        for spine in ax.spines.values():
            spine.set_color('#30363d')
            
        plt.tight_layout()
        return fig

if __name__ == "__main__":
    viz = VizEngine()
    sample_probs = {"y (↑)": 45, "y' (↓)": 15, "x (→)": 30, "x' (←)": 10}
    viz.plot_4d_forecast("AAPL", sample_probs)
