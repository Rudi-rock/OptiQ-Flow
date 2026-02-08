import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import List, Dict

def plot_optimization_trace(history: List[float]):
    """
    Plots the energy vs iteration curve.
    Returns a matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history, label='QAOA Optimization Trace', color='#6200EA', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Energy (Hamiltonian Value)')
    ax.set_title('QAOA Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

def plot_portfolio_allocation(assets: List[str], bitstring: str):
    """
    Plots the portfolio allocation based on the optimal bitstring.
    Returns a plotly figure.
    """
    # Bitstring is "1010" -> Asset 0 and Asset 2 selected
    selected = [assets[i] for i, bit in enumerate(bitstring) if bit == '1']
    
    if not selected:
        # Handle empty selection (should verify constraint prevents this)
        labels = ["None"]
        values = [1]
        title = "No Assets Selected (Constraint Violated?)"
    else:
        labels = selected
        values = [1] * len(selected) # Equal allocation
        title = f"Optimal Portfolio Allocation ({len(selected)} Assets)"

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
    fig.update_layout(title_text=title)
    return fig

def plot_results_comparison(classical_val: float, raw_val: float, rem_val: float, zne_val: float):
    """
    Plots a bar chart comparing Classical (Ground Truth) vs different Quantum modes.
    Returns a plotly figure.
    """
    categories = ['Classical (Truth)', 'Raw QAOA (Noisy)', 'Readout Mitigated', 'ZNE Mitigated']
    values = [classical_val, raw_val, rem_val, zne_val]
    colors = ['#00C853', '#D50000', '#FFAB00', '#2962FF'] # Green, Red, Amber, Blue
    
    fig = go.Figure(data=[go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f"{v:.4f}" for v in values],
        textposition='auto',
    )])
    
    min_val = min(values)
    max_val = max(values)
    padding = (max_val - min_val) * 0.1
    
    fig.update_layout(
        title_text='Quantum Optimization Performance Comparison',
        yaxis_title='Energy (Lower is Better)',
        yaxis_range=[min_val - padding, max_val + padding]
    )
    
    # Add horizontal line for classical truth
    fig.add_hline(y=classical_val, line_dash="dash", line_color="green", annotation_text="Ground Truth")
    
    return fig

def plot_error_heatmap(beta_vals, gamma_vals, energy_vals):
    """
    Heatmap of energy landscape (optional, for grid search).
    """
    fig = go.Figure(data=go.Heatmap(
        z=energy_vals,
        x=beta_vals,
        y=gamma_vals,
        colorscale='Viridis'
    ))
    fig.update_layout(
        title='QAOA Energy Landscape',
        xaxis_title='Beta',
        yaxis_title='Gamma'
    )
    return fig
