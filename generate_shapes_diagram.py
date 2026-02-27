import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_cylinder(ax, x, y, width, height, color):
    # Cylinder Top (Ellipse)
    top = patches.Ellipse((x + width/2, y + height), width, height*0.3, color=color, alpha=0.9, ec='white', lw=1.5)
    # Body (Rectangle)
    body = patches.Rectangle((x, y + height*0.15), width, height*0.85, color=color, alpha=0.7, ec='white', lw=1.5)
    # Bottom (Ellipse)
    bottom = patches.Ellipse((x + width/2, y + height*0.15), width, height*0.3, color=color, alpha=0.7, ec='white', lw=1.5)
    
    ax.add_patch(bottom)
    ax.add_patch(body)
    ax.add_patch(top)

def draw_diamond(ax, x, y, width, height, color):
    # Diamond coordinates
    points = [
        [x, y + height/2],     # Left
        [x + width/2, y + height], # Top
        [x + width, y + height/2], # Right
        [x + width/2, y]       # Bottom
    ]
    diamond = patches.Polygon(points, closed=True, color=color, alpha=0.8, ec='white', lw=2)
    ax.add_patch(diamond)

def draw_rectangle(ax, x, y, width, height, color):
    rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1", color=color, alpha=0.8, ec='white', lw=1.5)
    ax.add_patch(rect)

def create_shape_diagram():
    fig, ax = plt.subplots(figsize=(16, 6), facecolor='#0e1117')
    ax.set_facecolor('#0e1117')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Step Dimensions (Slimmed down to fit 6)
    w, h = 2.2, 1.6
    y_center = 2.5
    
    # --- PHASE 1: CYLINDER (BLUE) ---
    draw_cylinder(ax, 0.2, y_center - h/2, w, h, '#00d2ff')
    ax.text(0.2 + w/2, y_center, "PHASE 1\nData Ingestion\n(Market API)", ha='center', va='center', color='white', fontweight='bold', fontsize=8)

    # --- PHASE 2: RECTANGLE (ORANGE) ---
    draw_rectangle(ax, 2.7, y_center - h/2, w, h, '#ff9f1c') 
    ax.text(2.7 + w/2, y_center, "PHASE 2\nSignal\nProcessing", ha='center', va='center', color='white', fontweight='bold', fontsize=8)

    # --- PHASE 3: RECTANGLE (TEAL) ---
    draw_rectangle(ax, 5.2, y_center - h/2, w, h, '#2ec4b6')
    ax.text(5.2 + w/2, y_center, "PHASE 3\nLSTM Deep\nLearning", ha='center', va='center', color='white', fontweight='bold', fontsize=8)

    # --- PHASE 4: DIAMOND (RED) ---
    draw_diamond(ax, 7.8, y_center - h*0.8, w*1.1, h*1.6, '#ff4b4b')
    ax.text(7.8 + (w*1.1)/2, y_center, "PHASE 4\nMonte Carlo\nSimulation", ha='center', va='center', color='white', fontweight='bold', fontsize=8)

    # --- PHASE 5: RECTANGLE (PURPLE) ---
    draw_rectangle(ax, 10.7, y_center - h/2, w, h, '#7928ca')
    ax.text(10.7 + w/2, y_center, "PHASE 5\n4D Visual\nAnalytics", ha='center', va='center', color='white', fontweight='bold', fontsize=8)

    # --- PHASE 6: RECTANGLE (GOLD) ---
    draw_rectangle(ax, 13.5, y_center - h/2, w, h, '#ffc107')
    ax.text(13.5 + w/2, y_center, "PHASE 6\nCompetitive\nBenchmarking", ha='center', va='center', color='black', fontweight='bold', fontsize=8)

    # --- ARROWS ---
    # Adjusted connection points for 6 nodes
    connection_pairs = [(2.4, 2.7), (4.9, 5.2), (7.4, 7.8), (10.3, 10.7), (12.9, 13.5)]
    for start, end in connection_pairs:
        ax.annotate('', xy=(end, y_center), xytext=(start, y_center),
                    arrowprops=dict(arrowstyle='->', color='white', lw=1.5, mutation_scale=15))

    plt.title("End-to-End Hybrid LSTM–Monte Carlo System Architecture", fontsize=18, fontweight='bold', color='white', pad=40)
    
    plt.savefig("simplified_architecture_shapes.png", dpi=300, bbox_inches='tight', facecolor='#0e1117')
    print("Shape-accurate diagram saved as simplified_architecture_shapes.png")

if __name__ == "__main__":
    create_shape_diagram()
