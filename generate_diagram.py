import matplotlib.pyplot as plt
import networkx as nx

def generate_architecture_diagram():
    G = nx.DiGraph()

    # Define nodes with LSTM + Monte Carlo context
    nodes = {
        'User': 'User Input\n(Ticker)',
        'DataEngine': 'Data Engine\n(Fetch & Process)',
        'Yahoo': 'Yahoo Finance\n(API)',
        'Feature': 'Feature\nEngineering\n(RSI, SMA, Vol)',
        'Sequence': 'Sequence\nCreation\n(Windowing)',
        'LSTM_Model': 'LSTM Regressor\n(PyTorch)',
        'MonteCarlo': 'Monte Carlo\nEngine\n(1,000 Paths)',
        'VizEngine': 'Visualization\nEngine\n(4D Chart)',
        'Dashboard': 'Streamlit\nDashboard\n(Final Output)'
    }

    # Add edges
    G.add_edge(nodes['User'], nodes['DataEngine'])
    G.add_edge(nodes['DataEngine'], nodes['Yahoo'])
    G.add_edge(nodes['Yahoo'], nodes['DataEngine'], label='OHLCV Data')
    G.add_edge(nodes['DataEngine'], nodes['Feature'])
    G.add_edge(nodes['Feature'], nodes['Sequence'], label='Scaled Data')
    G.add_edge(nodes['Sequence'], nodes['LSTM_Model'], label='Time Series Tensor')
    G.add_edge(nodes['LSTM_Model'], nodes['MonteCarlo'], label='Expected Drift')
    G.add_edge(nodes['MonteCarlo'], nodes['VizEngine'], label='Probabilities')
    G.add_edge(nodes['VizEngine'], nodes['Dashboard'], label='Visual Analytics')

    plt.figure(figsize=(10, 12))
    pos = {
        nodes['User']: (0, 10),
        nodes['DataEngine']: (0, 8.5),
        nodes['Yahoo']: (2, 8.5),
        nodes['Feature']: (0, 7),
        nodes['Sequence']: (0, 5.5),
        nodes['LSTM_Model']: (0, 4),
        nodes['MonteCarlo']: (0, 2.5),
        nodes['VizEngine']: (0, 1),
        nodes['Dashboard']: (0, -0.5)
    }

    # Draw nodes
    nx.draw(G, pos, with_labels=True, 
            node_size=5000, 
            node_color='#aaddff', 
            font_size=9, 
            font_weight='bold', 
            arrowsize=15,
            edge_color='gray',
            node_shape='s',  # Square nodes for a flowchart look
            bbox=dict(facecolor='#aaddff', edgecolor='black', boxstyle='round,pad=0.2'))

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)

    plt.title("Hybrid LSTM–Monte Carlo Framework Architecture", fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("architecture_diagram.png", dpi=300, bbox_inches='tight')
    print("New LSTM architecture diagram saved as architecture_diagram.png")

if __name__ == "__main__":
    generate_architecture_diagram()
