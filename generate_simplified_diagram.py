import networkx as nx
import matplotlib.pyplot as plt

def generate_simplified_diagram():
    # 1. Create a Directed Graph
    G = nx.DiGraph()

    # 2. Define Nodes with specific labels
    nodes = {
        "Data": "1. Data Ingestion Layer\n(Yahoo Finance API)",
        "Features": "2. Feature Engineering &\nNormalization",
        "LSTM": "3. Hybrid LSTM\nNeural Core",
        "Simulator": "4. Monte Carlo\nStochastic Simulator\n(Decision Layer)",
        "Visuals": "5. 4D Visual Analytics\nDashboard"
    }

    # 3. Add Edges to define the flow (1 -> 2 -> 3 -> 4 -> 5)
    edges = [
        ("Data", "Features"),
        ("Features", "LSTM"),
        ("LSTM", "Simulator"),
        ("Simulator", "Visuals")
    ]

    G.add_nodes_from(nodes.keys())
    G.add_edges_from(edges)

    # 4. Set up the layout (Linear horizontal flow)
    pos = {
        "Data": (0, 0),
        "Features": (1, 0),
        "LSTM": (2, 0),
        "Simulator": (3, 0),
        "Visuals": (4, 0)
    }

    # 5. Create the Figure
    plt.figure(figsize=(15, 6), facecolor='#0e1117')
    ax = plt.gca()
    ax.set_facecolor('#0e1117')

    # 6. Draw Nodes
    # Using different colors/shapes logic in a simple way for visualization
    # Note: Diamond/Cylinder shapes are hard in basic networkx, 
    # so we use colors and bounding boxes to signify them.
    
    node_colors = ['#00d2ff', '#00ff88', '#00ff88', '#ff0055', '#7928ca'] # Blue, Green, Green, Red/Pink, Purple
    
    # Draw boxes manually for better control
    for i, node in enumerate(pos.keys()):
        x, y = pos[node]
        label = nodes[node]
        
        # Color the simulator (Step 4) differently to signify it's a decision diamond
        box_style = "round,pad=0.5"
        if node == "Simulator":
             box_style = "sawtooth,pad=0.8" # A rough "diamond-like" substitute
             
        plt.text(x, y, label, 
                 ha='center', va='center', 
                 fontsize=11, fontweight='bold', color='white',
                 bbox=dict(facecolor=node_colors[i], edgecolor='white', boxstyle=box_style, alpha=0.8))

    # 7. Draw Edges (Arrows)
    nx.draw_networkx_edges(G, pos, 
                           edgelist=edges, 
                           edge_color='white', 
                           arrows=True, 
                           arrowsize=20, 
                           width=2,
                           connectionstyle="arc3,rad=0.1")

    plt.title("Simplified Hybrid LSTM–Monte Carlo Architecture", fontsize=16, fontweight='bold', color='white', pad=30)
    plt.axis('off')
    
    # 8. Save the output
    plt.savefig("simplified_architecture.png", dpi=300, bbox_inches='tight', facecolor='#0e1117')
    print("Simplified architecture diagram saved as simplified_architecture.png")

if __name__ == "__main__":
    generate_simplified_diagram()
