import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from itertools import combinations
import scipy.linalg as la
from scipy.optimize import minimize

import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Qiskit Imports
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

# ==========================================
# COINDCX-STYLE THEME
# ==========================================
def apply_coindcx_theme():
    st.markdown("""
    <style>
    /* Global Dark Theme */
    .stApp {
        background-color: #0b0b0b;
        color: #e8e8e8;
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    
    /* Header Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background-color: #0b0b0b;
        padding: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #0b0b0b;
        color: #888;
        font-size: 16px;
        font-weight: 600;
        padding: 12px 24px;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0b0b0b;
        color: #4d9cff;
        border-bottom: 3px solid #4d9cff;
    }
    
    /* Card Containers */
    .category-card {
        background-color: #1a1a1a;
        border-radius: 16px;
        padding: 20px;
        margin: 12px 0;
    }
    
    .category-header {
        display: flex;
        align-items: center;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid #2a2a2a;
    }
    
    .category-icon {
        width: 40px;
        height: 40px;
        background-color: #2a2a2a;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        margin-right: 12px;
    }
    
    .category-title {
        flex: 1;
        font-size: 18px;
        font-weight: 700;
        color: #e8e8e8;
    }
    
    .category-subtitle {
        font-size: 13px;
        color: #888;
        margin-top: 2px;
    }
    
    .category-arrow {
        color: #888;
        font-size: 20px;
    }
    
    /* Asset List Items */
    .asset-item {
        display: flex;
        align-items: center;
        padding: 14px 0;
        border-bottom: 1px solid #2a2a2a;
    }
    
    .asset-item:last-child {
        border-bottom: none;
    }
    
    .asset-icon {
        width: 44px;
        height: 44px;
        border-radius: 50%;
        background: linear-gradient(135deg, #2a2a2a, #3a3a3a);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        font-weight: 700;
        margin-right: 14px;
        color: #4d9cff;
    }
    
    .asset-info {
        flex: 1;
    }
    
    .asset-name {
        font-size: 16px;
        font-weight: 600;
        color: #e8e8e8;
        margin-bottom: 2px;
    }
    
    .asset-symbol {
        font-size: 13px;
        color: #888;
    }
    
    .asset-price {
        text-align: right;
    }
    
    .asset-value {
        font-size: 16px;
        font-weight: 700;
        color: #e8e8e8;
        margin-bottom: 2px;
    }
    
    .asset-change {
        font-size: 14px;
        font-weight: 600;
    }
    
    .gain {
        color: #00d563;
    }
    
    .loss {
        color: #ff4d4d;
    }
    
    /* Control Panel */
    .control-panel {
        background-color: #1a1a1a;
        border-radius: 16px;
        padding: 20px;
        margin: 12px 0;
    }
    
    .control-section {
        margin-bottom: 20px;
    }
    
    .control-label {
        color: #888;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #4d9cff 0%, #3d7fcf 100%);
        color: #ffffff;
        font-weight: 700;
        font-size: 16px;
        padding: 14px;
        border-radius: 12px;
        border: none;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #5da9ff 0%, #4d8fdf 100%);
        box-shadow: 0 4px 12px rgba(77, 156, 255, 0.3);
    }
    
    /* Metrics */
    .metric-container {
        background-color: #1a1a1a;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }
    
    .metric-label {
        font-size: 12px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    
    .metric-value {
        font-size: 22px;
        font-weight: 700;
        font-family: monospace;
    }
    
    .metric-delta {
        font-size: 14px;
        font-weight: 600;
        margin-top: 4px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f0f0f;
        border-right: 1px solid #2a2a2a;
    }
    
    [data-testid="stSidebar"] > div {
        background-color: #0f0f0f;
    }
    
    /* Input widgets */
    .stMultiSelect, .stSlider, .stNumberInput, .stRadio {
        background-color: #1a1a1a;
    }
    
    /* Tabs content area */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 20px;
    }
    
    /* Charts */
    .chart-container {
        background-color: #1a1a1a;
        border-radius: 16px;
        padding: 20px;
        margin: 12px 0;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        color: #e8e8e8;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1a1a1a;
        border-radius: 12px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# CONFIGURATION
# ==========================================
@dataclass
class GlobalConfig:
    ASSETS: List[str] = None
    START_DATE: str = "2023-01-01"
    END_DATE: str = "2024-01-01"
    RISK_AVERSION: float = 0.5
    BUDGET: int = 1
    QAOA_DEPTH: int = 1
    SHOTS: int = 1024
    BACKEND_NAME: str = "aer_simulator"
    USE_REAL_BACKEND: bool = False
    SEED: int = 42

    def __post_init__(self):
        if self.ASSETS is None:
            self.ASSETS = ["AAPL", "MSFT", "GOOGL", "AMZN"]

config = GlobalConfig()

# ==========================================
# DATA LAYER
# ==========================================
def generate_synthetic_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n_days = len(dates)
    n_assets = len(tickers)
    np.random.seed(config.SEED)
    returns = np.random.normal(0.0005, 0.01, (n_days, n_assets))
    price_paths = 100 * np.cumprod(1 + returns, axis=0)
    return pd.DataFrame(price_paths, index=dates, columns=tickers)

def fetch_stock_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
        if data.empty or data.shape[1] != len(tickers):
            raise ValueError("Incomplete data")
        return data
    except Exception as e:
        print(f"Data fetch failed: {e}. Using synthetic.")
        return generate_synthetic_data(tickers, start_date, end_date)

def calculate_covariance_matrix(prices: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    returns = prices.pct_change().dropna()
    mu = returns.mean().values * 252
    sigma = returns.cov().values * 252
    return mu, sigma

# ==========================================
# QUBO FORMULATION
# ==========================================
class PortfolioOptimizationQUBO:
    def __init__(self, mu: np.ndarray, sigma: np.ndarray, risk_aversion: float, budget: int):
        self.mu = mu
        self.sigma = sigma
        self.q = risk_aversion 
        self.budget = budget
        self.num_assets = len(mu)
        self.qp = None
        
    def create_quadratic_program(self) -> QuadraticProgram:
        qp = QuadraticProgram()
        for i in range(self.num_assets):
            qp.binary_var(name=f"x_{i}")
        quadratic = {}
        for i in range(self.num_assets):
            for j in range(self.num_assets):
                quadratic[(f"x_{i}", f"x_{j}")] = self.q * self.sigma[i, j]
        linear = {}
        for i in range(self.num_assets):
            linear[f"x_{i}"] = -(1 - self.q) * self.mu[i]
        qp.minimize(linear=linear, quadratic=quadratic)
        linear_constraint = {f"x_{i}": 1 for i in range(self.num_assets)}
        qp.linear_constraint(linear=linear_constraint, sense="==", rhs=self.budget, name="budget")
        self.qp = qp
        return qp
    
    def to_ising(self) -> Tuple[SparsePauliOp, float]:
        if self.qp is None:
            self.create_quadratic_program()
        conv = QuadraticProgramToQubo()
        qubo = conv.convert(self.qp)
        return qubo.to_ising()

# ==========================================
# CLASSICAL SOLVER
# ==========================================
@dataclass
class ClassicalResult:
    indices: list
    bitstring: str
    value: float
    ret: float
    volatility: float
    sharpe: float

class ClassicalPortfolioSolver:
    def __init__(self, mu: np.ndarray, sigma: np.ndarray, risk_aversion: float, budget: int):
        self.mu = mu
        self.sigma = sigma
        self.q = risk_aversion
        self.budget = budget
        self.num_assets = len(mu)

    def solve(self) -> ClassicalResult:
        indices = list(range(self.num_assets))
        best_value = float('inf')
        best_indices = []
        for combo in combinations(indices, self.budget):
            x = np.zeros(self.num_assets)
            x[list(combo)] = 1.0
            risk = x.T @ self.sigma @ x
            return_val = self.mu.T @ x
            obj_value = self.q * risk - (1 - self.q) * return_val
            if obj_value < best_value:
                best_value = obj_value
                best_indices = list(combo)
        
        best_x = np.zeros(self.num_assets)
        best_x[best_indices] = 1.0
        final_risk = np.sqrt(best_x.T @ self.sigma @ best_x)
        final_return = self.mu.T @ best_x
        sharpe = final_return / final_risk if final_risk > 1e-6 else 0.0
        bitstring = "".join(["1" if i in best_indices else "0" for i in range(self.num_assets)])
        
        return ClassicalResult(best_indices, bitstring, best_value, final_return, final_risk, sharpe)

# ==========================================
# QUANTUM BACKEND
# ==========================================
def get_noisy_simulator():
    noise_model = NoiseModel()
    error_1q = depolarizing_error(0.001, 1)
    error_2q = depolarizing_error(0.01, 2)
    noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3', 'rz', 'sx', 'x'])
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz', 'ecr'])
    p1_0 = 0.02
    readout_error = ReadoutError([[1 - p1_0, p1_0], [p1_0, 1 - p1_0]])
    noise_model.add_all_qubit_readout_error(readout_error)
    return AerSimulator(noise_model=noise_model)

def get_backend(use_real_backend: bool = False):
    if use_real_backend:
        try:
            service = QiskitRuntimeService()
            return service.backend("ibm_brisbane")
        except:
            return get_noisy_simulator()
    return get_noisy_simulator()

# ==========================================
# ERROR MITIGATION
# ==========================================
class ReadoutMitigator:
    def __init__(self, backend, num_qubits: int):
        self.backend = backend
        self.num_qubits = num_qubits
        self.calibration_matrices = []
        
    def calibrate(self):
        qc0 = QuantumCircuit(self.num_qubits, self.num_qubits)
        qc0.measure(range(self.num_qubits), range(self.num_qubits))
        qc1 = QuantumCircuit(self.num_qubits, self.num_qubits)
        qc1.x(range(self.num_qubits))
        qc1.measure(range(self.num_qubits), range(self.num_qubits))
        
        t_circuits = transpile([qc0, qc1], self.backend)
        result = self.backend.run(t_circuits, shots=2048).result()
        counts0 = result.get_counts(0)
        counts1 = result.get_counts(1)
        
        self.calibration_matrices = []
        for i in range(self.num_qubits):
            p0_given_0 = self._get_marginal_prob(counts0, i, '0')
            p1_given_1 = self._get_marginal_prob(counts1, i, '1')
            M = np.array([[p0_given_0, 1 - p1_given_1], [1 - p0_given_0, p1_given_1]])
            self.calibration_matrices.append(M)

    def _get_marginal_prob(self, counts, qubit_index, target_state):
        total = sum(counts.values())
        match = 0
        for bitstring, count in counts.items():
            if bitstring[self.num_qubits - 1 - qubit_index] == target_state:
                match += count
        return match / total

    def apply_correction(self, raw_counts):
        dim = 2**self.num_qubits
        vec = np.zeros(dim)
        total = sum(raw_counts.values())
        for b, c in raw_counts.items():
            vec[int(b, 2)] = c / total
            
        M_total_inv = np.array([[1.0]])
        for M in self.calibration_matrices:
            try:
                M_inv = la.inv(M)
            except:
                M_inv = np.eye(2)
            M_total_inv = np.kron(M_inv, M_total_inv)
            
        corrected = np.clip(M_total_inv @ vec, 0, 1)
        corrected /= np.sum(corrected)
        
        return {format(i, f'0{self.num_qubits}b'): val for i, val in enumerate(corrected) if val > 1e-6}

def global_folding(circuit: QuantumCircuit, scale: float) -> QuantumCircuit:
    k = int(round((max(1, scale) - 1) / 2))
    if k == 0: return circuit.copy()
    
    unitary = QuantumCircuit(circuit.num_qubits)
    for instr in circuit.data:
        if instr.operation.name != "measure":
            unitary.append(instr.operation, instr.qubits, instr.clbits)
            
    folded = unitary.copy()
    inv = unitary.inverse()
    for _ in range(k):
        folded.barrier()
        folded = folded.compose(inv)
        folded.barrier()
        folded = folded.compose(unitary)
        
    folded.measure_all()
    return folded

class ZNEMitigator:
    def __init__(self, scales): self.scales = scales
    def extrapolate(self, energies):
        coeffs = np.polyfit(self.scales, energies, 1)
        return coeffs[1]

# ==========================================
# QAOA EXECUTOR
# ==========================================
@dataclass
class QAOAResult:
    optimal_params: List[float]
    optimal_value: float
    history: List[float]
    circuit: QuantumCircuit

class RawQAOARunner:
    def __init__(self, hamiltonian, backend, reps=1):
        self.H = hamiltonian
        self.backend = backend
        self.reps = reps
        self.ansatz = QAOAAnsatz(cost_operator=hamiltonian, reps=reps)
        self.estimator = AerEstimator(run_options={"shots": config.SHOTS, "seed": config.SEED}, backend_options={"method": "density_matrix", "noise_model": backend.options.noise_model})
        self.history = []

    def run(self, maxiter=25):
        def cost(params):
            val = self.estimator.run(self.ansatz, self.H, parameter_values=params).result().values[0]
            self.history.append(val)
            return val
        
        res = minimize(cost, np.random.uniform(-np.pi, np.pi, 2*self.reps), method='COBYLA', options={'maxiter': maxiter})
        bound = self.ansatz.assign_parameters(res.x)
        bound.measure_all()
        return QAOAResult(res.x.tolist(), res.fun, self.history, bound)

class MitigatedExecutor:
    def __init__(self, backend, hamiltonian):
        self.backend = backend
        self.H = hamiltonian
        self.rem = None

    def execute(self, circuit, scales=[1.0, 3.0, 5.0]):
        if not self.rem:
            self.rem = ReadoutMitigator(self.backend, circuit.num_qubits)
            self.rem.calibrate()
            
        energies = []
        raw_res = None
        rem_res = None
        
        for s in scales:
            qc = global_folding(circuit, s)
            if qc.num_clbits == 0: qc.measure_all()
            cts = self.backend.run(transpile(qc, self.backend), shots=4096).result().get_counts()
            mit_cts = self.rem.apply_correction(cts)
            en = self._compute_energy(mit_cts)
            energies.append(en)
            if s == 1.0:
                raw_res = self._compute_energy(cts)
                rem_res = en
                
        zne = ZNEMitigator(scales).extrapolate(energies)
        return {"raw": raw_res, "rem": rem_res, "zne": zne, "hist": energies}

    def _compute_energy(self, counts):
        E = 0
        total = sum(counts.values())
        for pauli, coeff in self.H.label_iter():
            term = 0
            for b, c in counts.items():
                prob = c / total
                parity = sum(1 for i, char in enumerate(pauli) if char == 'Z' and b[i] == '1')
                term += ((-1)**parity) * prob
            E += term * coeff.real
        return E

# ==========================================
# UI COMPONENTS
# ==========================================
def render_asset_item(symbol, name, value, change_pct, is_gain):
    """Render individual asset item in CoinDCX list style"""
    change_class = "gain" if is_gain else "loss"
    change_symbol = "+" if is_gain else ""
    
    # Get first 2 letters for icon
    icon_text = symbol[:2].upper()
    
    st.markdown(f"""
    <div class="asset-item">
        <div class="asset-icon">{icon_text}</div>
        <div class="asset-info">
            <div class="asset-name">{name}</div>
            <div class="asset-symbol">{symbol}</div>
        </div>
        <div class="asset-price">
            <div class="asset-value">{value}</div>
            <div class="asset-change {change_class}">{change_symbol}{change_pct:.2f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_energy_chart(history):
    """Create QAOA convergence chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(history))),
        y=history,
        mode='lines+markers',
        name='Energy',
        line=dict(color='#4d9cff', width=2),
        marker=dict(size=5, color='#4d9cff'),
        fill='tozeroy',
        fillcolor='rgba(77, 156, 255, 0.1)'
    ))
    
    fig.update_layout(
        title='QAOA Energy Convergence',
        title_font=dict(size=14, color='#e8e8e8'),
        xaxis_title='Iteration',
        yaxis_title='Energy',
        plot_bgcolor='#0b0b0b',
        paper_bgcolor='#1a1a1a',
        font=dict(color='#e8e8e8'),
        xaxis=dict(gridcolor='#2a2a2a', showgrid=True, zeroline=False),
        yaxis=dict(gridcolor='#2a2a2a', showgrid=True, zeroline=False),
        hovermode='x unified',
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

def create_comparison_chart(c_val, raw, rem, zne):
    """Create method comparison chart"""
    methods = ['Classical', 'Raw QAOA', 'Readout Mit.', 'ZNE Mit.']
    values = [c_val, raw, rem, zne]
    
    # Color based on improvement
    colors = []
    for v in values:
        if v < c_val:
            colors.append('#00d563')
        elif v > c_val:
            colors.append('#ff4d4d')
        else:
            colors.append('#888888')
    colors[0] = '#4d9cff'  # Classical baseline
    
    fig = go.Figure(data=[go.Bar(
        x=methods,
        y=values,
        marker_color=colors,
        text=[f'{v:.6f}' for v in values],
        textposition='outside',
        textfont=dict(color='#e8e8e8', size=11)
    )])
    
    fig.update_layout(
        title='Method Comparison',
        title_font=dict(size=14, color='#e8e8e8'),
        yaxis_title='Energy',
        plot_bgcolor='#0b0b0b',
        paper_bgcolor='#1a1a1a',
        font=dict(color='#e8e8e8'),
        xaxis=dict(gridcolor='#2a2a2a', showgrid=False),
        yaxis=dict(gridcolor='#2a2a2a', showgrid=True, zeroline=False),
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=False
    )
    
    return fig

# ==========================================
# MAIN APP
# ==========================================
def main():
    st.set_page_config(
        page_title="OptiQ-Flow",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    apply_coindcx_theme()
    
    # Session state
    if 'status' not in st.session_state:
        st.session_state.status = "idle"
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    # Title
    st.markdown('<h1 style="color: #e8e8e8; font-size: 28px; margin-bottom: 10px;">⚛️ OptiQ-Flow</h1>', unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Overview", "Categories", "Results"])
    
    # Sidebar Controls
    with st.sidebar:
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        
        st.markdown('<div class="control-section">', unsafe_allow_html=True)
        st.markdown('<div class="control-label">Asset Selection</div>', unsafe_allow_html=True)
        tickers = st.multiselect(
            "Select Assets",
            ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"],
            default=["AAPL", "MSFT", "GOOGL", "AMZN"],
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="control-section">', unsafe_allow_html=True)
        st.markdown('<div class="control-label">Portfolio Parameters</div>', unsafe_allow_html=True)
        budget = st.number_input("Assets to Select", 1, len(tickers) if tickers else 4, 2)
        risk = st.slider("Risk Aversion λ", 0.0, 1.0, 0.5, 0.05)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="control-section">', unsafe_allow_html=True)
        st.markdown('<div class="control-label">Quantum Backend</div>', unsafe_allow_html=True)
        backend_type = st.radio(
            "Backend Type",
            ["Simulator", "Real Hardware"],
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        run_solver = st.button("🚀 EXECUTE SOLVER")
    
    # Execute solver
    if run_solver and tickers:
        st.session_state.status = "running"
        
        with st.spinner("Executing quantum optimization..."):
            # Data fetch
            df = fetch_stock_data(tickers, config.START_DATE, config.END_DATE)
            mu, sigma = calculate_covariance_matrix(df)
            
            # Classical solver
            c_solver = ClassicalPortfolioSolver(mu, sigma, risk, budget)
            c_res = c_solver.solve()
            
            # Quantum setup
            qubo = PortfolioOptimizationQUBO(mu, sigma, risk, budget)
            op, _ = qubo.to_ising()
            backend = get_backend(backend_type == "Real Hardware")
            
            # QAOA
            runner = RawQAOARunner(op, backend)
            q_res = runner.run()
            
            # Mitigation
            mitigator = MitigatedExecutor(backend, op)
            m_res = mitigator.execute(q_res.circuit)
            
            # Store results
            st.session_state.results = {
                'classical': c_res,
                'qaoa': q_res,
                'mitigated': m_res,
                'tickers': tickers,
                'mu': mu,
                'sigma': sigma
            }
            st.session_state.status = "done"
            st.rerun()
    
    # Tab 1: Overview
    with tab1:
        if st.session_state.results:
            results = st.session_state.results
            c_res = results['classical']
            m_res = results['mitigated']
            
            st.markdown('<div class="category-card">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: #e8e8e8; margin-bottom: 20px;">Portfolio Summary</h3>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Sharpe Ratio</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value gain">{c_res.sharpe:.4f}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Expected Return</div>', unsafe_allow_html=True)
                return_class = "gain" if c_res.ret > 0 else "loss"
                st.markdown(f'<div class="metric-value {return_class}">{c_res.ret*100:.2f}%</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Volatility</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value" style="color: #e8e8e8;">{c_res.volatility*100:.2f}%</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Selected Assets
            st.markdown('<div class="category-card">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: #e8e8e8; margin-bottom: 20px;">Selected Portfolio</h3>', unsafe_allow_html=True)
            
            selected_tickers = results['tickers']
            mu = results['mu']
            
            for idx in c_res.indices:
                ticker = selected_tickers[idx]
                annual_return = mu[idx] * 100
                is_gain = annual_return > 0
                
                render_asset_item(
                    ticker,
                    ticker,
                    f"{annual_return:.2f}%",
                    annual_return,
                    is_gain
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Configure parameters and click EXECUTE SOLVER to begin")
    
    # Tab 2: Categories
    with tab2:
        if st.session_state.results:
            results = st.session_state.results
            selected_tickers = results['tickers']
            mu = results['mu']
            sigma = results['sigma']
            c_res = results['classical']
            m_res = results['mitigated']
            
            # Large Cap - All Assets
            st.markdown(f"""
            <div class="category-card">
                <div class="category-header">
                    <div class="category-icon">📊</div>
                    <div style="flex: 1;">
                        <div class="category-title">Portfolio Assets</div>
                        <div class="category-subtitle">All available assets by return</div>
                    </div>
                    <div class="category-arrow">›</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Sort by return
            asset_data = [(i, selected_tickers[i], mu[i], np.sqrt(sigma[i, i])) 
                         for i in range(len(selected_tickers))]
            asset_data_sorted = sorted(asset_data, key=lambda x: x[2], reverse=True)
            
            for idx, ticker, ret, vol in asset_data_sorted:
                annual_return = ret * 100
                is_gain = annual_return > 0
                render_asset_item(
                    ticker,
                    ticker,
                    f"{annual_return:.2f}%",
                    annual_return,
                    is_gain
                )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Top Gainers - Quantum Advantage
            st.markdown(f"""
            <div class="category-card">
                <div class="category-header">
                    <div class="category-icon">🚀</div>
                    <div style="flex: 1;">
                        <div class="category-title">Quantum Advantage</div>
                        <div class="category-subtitle">Assets where quantum beats classical</div>
                    </div>
                    <div class="category-arrow">›</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Calculate quantum improvement per asset
            quantum_improvements = []
            for idx in c_res.indices:
                ticker = selected_tickers[idx]
                # Simplified: show ZNE improvement
                improvement = ((c_res.value - m_res['zne']) / abs(c_res.value)) * 100
                if improvement > 0:
                    quantum_improvements.append((idx, ticker, improvement))
            
            quantum_improvements_sorted = sorted(quantum_improvements, key=lambda x: x[2], reverse=True)
            
            if quantum_improvements_sorted:
                for idx, ticker, improvement in quantum_improvements_sorted:
                    render_asset_item(
                        ticker,
                        f"{ticker} (Quantum Optimized)",
                        f"+{improvement:.2f}%",
                        improvement,
                        True
                    )
            else:
                st.markdown('<div style="padding: 20px; color: #888;">No quantum advantage detected</div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # High Volatility
            st.markdown(f"""
            <div class="category-card">
                <div class="category-header">
                    <div class="category-icon">⚡</div>
                    <div style="flex: 1;">
                        <div class="category-title">High Volatility</div>
                        <div class="category-subtitle">Assets with highest risk</div>
                    </div>
                    <div class="category-arrow">›</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Sort by volatility
            volatility_data = [(i, selected_tickers[i], np.sqrt(sigma[i, i])) 
                              for i in range(len(selected_tickers))]
            volatility_sorted = sorted(volatility_data, key=lambda x: x[2], reverse=True)[:5]
            
            for idx, ticker, vol in volatility_sorted:
                vol_pct = vol * 100
                render_asset_item(
                    ticker,
                    ticker,
                    f"{vol_pct:.2f}%",
                    vol_pct,
                    False
                )
            
        else:
            st.info("Run solver to see asset categories")
    
    # Tab 3: Results
    with tab3:
        if st.session_state.results:
            results = st.session_state.results
            c_res = results['classical']
            q_res = results['qaoa']
            m_res = results['mitigated']
            
            # Energy comparison
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(
                create_comparison_chart(c_res.value, m_res['raw'], m_res['rem'], m_res['zne']),
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Convergence
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(create_energy_chart(q_res.history), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Detailed metrics
            st.markdown('<div class="category-card">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: #e8e8e8; margin-bottom: 20px;">Energy Metrics</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Classical Energy</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value" style="color: #4d9cff;">{c_res.value:.6f}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Readout Mitigated</div>', unsafe_allow_html=True)
                rem_better = m_res['rem'] < c_res.value
                rem_class = "gain" if rem_better else "loss"
                st.markdown(f'<div class="metric-value {rem_class}">{m_res["rem"]:.6f}</div>', unsafe_allow_html=True)
                delta = m_res['rem'] - c_res.value
                delta_class = "gain" if delta < 0 else "loss"
                st.markdown(f'<div class="metric-delta {delta_class}">{delta:+.6f}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Raw QAOA</div>', unsafe_allow_html=True)
                raw_better = m_res['raw'] < c_res.value
                raw_class = "gain" if raw_better else "loss"
                st.markdown(f'<div class="metric-value {raw_class}">{m_res["raw"]:.6f}</div>', unsafe_allow_html=True)
                delta = m_res['raw'] - c_res.value
                delta_class = "gain" if delta < 0 else "loss"
                st.markdown(f'<div class="metric-delta {delta_class}">{delta:+.6f}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">ZNE Mitigated</div>', unsafe_allow_html=True)
                zne_better = m_res['zne'] < c_res.value
                zne_class = "gain" if zne_better else "loss"
                st.markdown(f'<div class="metric-value {zne_class}">{m_res["zne"]:.6f}</div>', unsafe_allow_html=True)
                delta = m_res['zne'] - c_res.value
                delta_class = "gain" if delta < 0 else "loss"
                st.markdown(f'<div class="metric-delta {delta_class}">{delta:+.6f}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Results table
            st.markdown('<div class="category-card">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: #e8e8e8; margin-bottom: 20px;">Detailed Results</h3>', unsafe_allow_html=True)
            
            summary_df = pd.DataFrame({
                'Method': ['Classical', 'Raw QAOA', 'Readout Mitigation', 'ZNE Mitigation'],
                'Energy': [c_res.value, m_res['raw'], m_res['rem'], m_res['zne']],
                'Delta vs Classical': [0.0, 
                                      m_res['raw'] - c_res.value, 
                                      m_res['rem'] - c_res.value, 
                                      m_res['zne'] - c_res.value],
                'Status': ['Baseline',
                          '✓ Better' if m_res['raw'] < c_res.value else '✗ Worse',
                          '✓ Better' if m_res['rem'] < c_res.value else '✗ Worse',
                          '✓ Better' if m_res['zne'] < c_res.value else '✗ Worse']
            })
            
            st.dataframe(
                summary_df.style.format({
                    'Energy': '{:.6f}',
                    'Delta vs Classical': '{:+.6f}'
                }),
                use_container_width=True,
                hide_index=True
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Run solver to see results")

if __name__ == "__main__":
    main()

