import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.global_config import config
from data.fetch_data import fetch_stock_data
from data.portfolio_data import calculate_covariance_matrix
from qubo.qubo_formulation import PortfolioOptimizationQUBO
from classical.classical_baseline import ClassicalPortfolioSolver
from quantum.backend import get_backend
from quantum.qaoa_raw import RawQAOARunner
from quantum.qaoa_mitigated import MitigatedQAOAExecutor
from visualization.visualization_helpers import (
    plot_optimization_trace, 
    plot_portfolio_allocation, 
    plot_results_comparison
)

# Page Config
st.set_page_config(
    page_title="OptiQ-Flow",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: #0E1117;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        background: linear-gradient(90deg, #FF4B4B, #FF9051);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/51/Qiskit-Logo.svg/1200px-Qiskit-Logo.svg.png", width=100)
    st.title("Settings")
    
    st.subheader("Reference Data")
    default_assets = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"]
    selected_assets = st.multiselect("Select Assets", default_assets, default=["AAPL", "MSFT", "GOOGL", "AMZN"])
    
    st.subheader("Portfolio Parameters")
    budget = st.number_input("Budget (Num Assets)", min_value=1, max_value=len(selected_assets), value=2)
    risk_aversion = st.slider("Risk Aversion (lambda)", 0.0, 1.0, 0.5, 0.1)
    
    st.subheader("Quantum Config")
    backend_choice = st.radio("Backend", ["Noisy Simulator", "Real Hardware (Experimental)"])
    qaoa_depth = st.slider("QAOA Depth (p)", 1, 3, 1)
    use_zne = st.checkbox("Enable ZNE Mitigation", value=True)
    
    run_btn = st.button("RUN OPTIMIZATION")

# Main Content
st.title("OptiQ-Flow: Quantum Portfolio Optimization")
st.markdown("### Hybrid Classical-Quantum Solver with Error Mitigation")

# Logic
if run_btn:
    if len(selected_assets) < 2:
        st.error("Please select at least 2 assets.")
        st.stop()
        
    with st.spinner("Initializing Pipeline..."):
        # 1. Data
        st.info("Fetching Financial Data...", icon="📈")
        try:
            df = fetch_stock_data(selected_assets, config.START_DATE, config.END_DATE)
            mu, sigma = calculate_covariance_matrix(df)
        except Exception as e:
            st.error(f"Data Error: {e}")
            st.stop()
            
    # Layout using Tabs
    tab1, tab2, tab3 = st.tabs(["Analysis", "Quantum Execution", "Results"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Stock Prices (Synthetis/Real)")
            st.line_chart(df)
        with col2:
            st.subheader("Covariance Matrix")
            st.write(sigma)
            
    # 2. Solver
    with st.spinner("Solving..."):
        # Classical
        classical_solver = ClassicalPortfolioSolver(mu, sigma, risk_aversion, budget)
        c_res = classical_solver.solve()
        
        # Quantum
        qubo = PortfolioOptimizationQUBO(mu, sigma, risk_aversion, budget)
        op, offset = qubo.to_ising()
        
        use_real = (backend_choice == "Real Hardware (Experimental)")
        backend = get_backend(use_real_backend=use_real)
        
        # 4a. Optimize Params (Raw)
        raw_runner = RawQAOARunner(op, backend, reps=qaoa_depth)
        raw_res = raw_runner.run(maxiter=25)
        
        # 4b. Mitigated Execution
        mit_executor = MitigatedQAOAExecutor(backend, op)
        scales = [1.0, 3.0, 5.0] if use_zne else [1.0]
        mit_res = mit_executor.execute(raw_res.circuit, zne_scales=scales)
        
    # Tab 2: Quantum Details
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Optimization Trace (Energy)")
            fig_trace = plot_optimization_trace(raw_res.history)
            st.pyplot(fig_trace)
        with col2:
            st.subheader("ZNE Extrapolation")
            if use_zne:
                st.write(f"Energies at scales {scales}: {mit_res.get('zne_history', [])}")
                st.line_chart(pd.DataFrame({'Scale': scales, 'Energy': mit_res['zne_history']}).set_index('Scale'))
            else:
                st.info("ZNE Disabled.")
                
    # Tab 3: Final Results
    with tab3:
        st.success("Optimization Complete!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Classical Baseline", f"{c_res.value:.4f}")
        with col2:
            st.metric("Raw QAOA", f"{mit_res['raw_value']:.4f}", delta=f"{mit_res['raw_value'] - c_res.value:.4f} err", delta_color="inverse")
        with col3:
            st.metric("Mitigated QAOA", f"{mit_res['zne_value']:.4f}", delta=f"{mit_res['zne_value'] - c_res.value:.4f} err", delta_color="inverse")
            
        st.subheader("Performance Comparison")
        fig_comp = plot_results_comparison(
            c_res.value, 
            mit_res['raw_value'], 
            mit_res['rem_value'], 
            mit_res['zne_value']
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        
        st.subheader("Optimal Portfolio")
        st.write(f"Classical Bitstring: {c_res.bitstring}")
        
        fig_pie = plot_portfolio_allocation(selected_assets, c_res.bitstring)
        st.plotly_chart(fig_pie)

else:
    st.info("Select assets and click RUN to start.")
