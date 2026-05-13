"""
Attention extraction, heatmap visualization, and propagation network analysis.
For iTransformer post-hoc analysis — core contribution of the paper.
"""
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

from config import DEVICE, N_VARIATES, CORE_VARIATES, RESULT_DIR


def extract_attention_matrix(model, test_loader, device=DEVICE, n_variates=N_VARIATES):
    """
    Extract average attention matrix across all test batches.
    v4: passes temporal context to model for accurate attention extraction.
    Returns: (n_variates, n_variates) numpy array.
    """
    from config import N_TEMPORAL

    model.eval()
    all_attn = []

    for batch in test_loader:
        x = batch["x"].to(device)
        x_price    = x[:, :, :n_variates]                             # (B, T, 33)
        x_temporal = x[:, :, n_variates:n_variates + N_TEMPORAL]      # (B, T, 15)
        attn = model.get_attention_matrix(x_price, x_temporal)        # (33, 33)
        all_attn.append(attn)

    avg_attn = np.mean(all_attn, axis=0)
    return avg_attn


def plot_attention_heatmap(attn_matrix, variate_names=None, save_path=None):
    """
    Plot 33×33 attention heatmap grouped by commodity.
    Groups: beras_cons(6), bawang_cons(6), cabai_cons(6),
            beras_prod(5), bawang_prod(5), cabai_prod(5)
    """
    if variate_names is None:
        variate_names = CORE_VARIATES
    if save_path is None:
        save_path = RESULT_DIR / "attention_heatmap.png"

    # Shortened labels for readability
    short_names = [v.replace('_cons', '©').replace('_prod', 'ⓟ') for v in variate_names]

    fig, ax = plt.subplots(figsize=(18, 15))
    sns.heatmap(
        attn_matrix, annot=False, cmap='YlOrRd',
        xticklabels=short_names, yticklabels=short_names,
        ax=ax, cbar_kws={'shrink': 0.8, 'label': 'Attention Weight'},
        linewidths=0.1,
    )

    # Draw commodity group separators
    #  Consumer: beras(0-5), bawang(6-11), cabai(12-17)
    #  Producer: beras(18-22), bawang(23-27), cabai(28-32)
    boundaries = [6, 12, 18, 23, 28]
    for b in boundaries:
        ax.axhline(b, color='white', linewidth=2.5)
        ax.axvline(b, color='white', linewidth=2.5)

    # Add group labels
    group_labels = ['Beras\nCons', 'Bawang\nCons', 'Cabai\nCons',
                    'Beras\nProd', 'Bawang\nProd', 'Cabai\nProd']
    group_centers = [3, 9, 15, 20.5, 25.5, 30.5]
    for gc, gl in zip(group_centers, group_labels):
        ax.text(-2.5, gc, gl, ha='center', va='center', fontsize=8,
                fontweight='bold', color='#333')

    ax.set_title(
        'iTransformer Variate Attention Matrix\n'
        '(Cross-Variate Propagation Structure — Core Contribution)',
        fontsize=14, pad=20, fontweight='bold',
    )
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved attention heatmap: {save_path}")


def build_propagation_network(attn_matrix, variate_names=None, threshold_quantile=0.75):
    """
    Convert attention matrix to directed propagation network.
    Edge (src→dst) exists if attn[dst, src] > threshold (src influences dst).

    Returns: (G, out_centrality, in_centrality)
    """
    if variate_names is None:
        variate_names = CORE_VARIATES

    threshold = np.quantile(attn_matrix, threshold_quantile)
    G = nx.DiGraph()
    G.add_nodes_from(variate_names)

    for i, src in enumerate(variate_names):
        for j, dst in enumerate(variate_names):
            if i != j and attn_matrix[j, i] > threshold:
                G.add_edge(src, dst, weight=float(attn_matrix[j, i]))

    out_centrality = nx.out_degree_centrality(G)
    in_centrality  = nx.in_degree_centrality(G)

    print(f"\n  Propagation Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Threshold (q={threshold_quantile}): {threshold:.4f}")
    print("\n  Top 5 most INFLUENTIAL variates (high out-degree centrality):")
    for v, c in sorted(out_centrality.items(), key=lambda x: -x[1])[:5]:
        print(f"    {v}: {c:.4f}")
    print("\n  Top 5 most AFFECTED variates (high in-degree centrality):")
    for v, c in sorted(in_centrality.items(), key=lambda x: -x[1])[:5]:
        print(f"    {v}: {c:.4f}")

    return G, out_centrality, in_centrality


def plot_propagation_graph(G, save_path=None):
    """Plot the propagation network graph."""
    if save_path is None:
        save_path = RESULT_DIR / "propagation_network.png"

    fig, ax = plt.subplots(figsize=(16, 12))

    # Color nodes by type: consumer=blue, producer=green
    node_colors = []
    for node in G.nodes():
        if '_cons' in node:
            node_colors.append('#4A90D9')
        else:
            node_colors.append('#50C878')

    # Size by out-degree
    out_deg = dict(G.out_degree())
    max_deg = max(out_deg.values()) if out_deg else 1
    node_sizes = [300 + 1500 * (out_deg[n] / max_deg) for n in G.nodes()]

    pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)

    # Draw edges with weight-based alpha
    edges = G.edges(data=True)
    if edges:
        weights = [d['weight'] for _, _, d in edges]
        max_w = max(weights) if weights else 1
        edge_alphas = [0.2 + 0.6 * (w / max_w) for w in weights]
        for (u, v, d), alpha in zip(edges, edge_alphas):
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], alpha=alpha,
                edge_color='#666', arrows=True, arrowsize=12,
                width=0.5 + 2.0 * (d['weight'] / max_w), ax=ax,
            )

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax, alpha=0.9)

    # Short labels
    labels = {n: n.replace('_cons', '©').replace('_prod', 'ⓟ') for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight='bold', ax=ax)

    ax.set_title('Price Spike Propagation Network\n(iTransformer Attention-Based)',
                 fontsize=14, fontweight='bold', pad=15)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4A90D9',
               markersize=12, label='Consumer Price'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#50C878',
               markersize=12, label='Producer Price'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved propagation network: {save_path}")


def save_centrality_report(out_centrality, in_centrality, save_path=None):
    """Save centrality metrics to CSV for the paper."""
    if save_path is None:
        save_path = RESULT_DIR / "centrality_report.csv"

    rows = []
    for v in out_centrality:
        rows.append({
            "variate": v,
            "out_centrality": round(out_centrality[v], 4),
            "in_centrality": round(in_centrality[v], 4),
            "net_influence": round(out_centrality[v] - in_centrality[v], 4),
        })
    df = pd.DataFrame(rows).sort_values("net_influence", ascending=False)
    df.to_csv(save_path, index=False)
    print(f"  [OK] Saved centrality report: {save_path}")
    return df
