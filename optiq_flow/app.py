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
# TRADING TERMINAL STYLING
# ==========================================
def apply_trading_terminal_style():
    st.markdown("""
    <style>
    /* Global Dark Theme */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Header Bar */
    .terminal-header {
        background: linear-gradient(90deg, #1a1f2e 0%, #0f1419 100%);
        padding: 15px 20px;
        border-bottom: 2px solid #00ff88;
        margin-bottom: 20px;
        border-radius: 8px;
    }
    
    .terminal-title {
        font-size: 28px;
        font-weight: 700;
        color: #00ff88;
        font-family: 'Courier New', monospace;
        letter-spacing: 2px;
    }
    
    .status-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
        font-family: monospace;
        margin-left: 15px;
    }
    
    .status-idle {
        background-color: #404040;
        color: #ffffff;
    }
    
    .status-running {
        background-color: #ff9500;
        color: #000000;
        animation: pulse 1.5s infinite;
    }
    
    .status-done {
        background-color: #00ff88;
        color: #000000;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    /* Control Panel */
    .control-panel {
        background-color: #1a1f2e;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #2d3748;
    }
    
    .control-section {
        margin-bottom: 20px;
        padding-bottom: 15px;
        border-bottom: 1px solid #2d3748;
    }
    
    .control-label {
        color: #00ff88;
        font-weight: 600;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    
    /* Metrics Display */
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #0f1419 100%);
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #00ff88;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 11px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #00ff88;
        font-family: 'Courier New', monospace;
    }
    
    .metric-value.negative {
        color: #ff4444;
    }
    
    .metric-delta {
        font-size: 14px;
        margin-top: 5px;
    }
    
    .delta-positive {
        color: #00ff88;
    }
    
    .delta-negative {
        color: #ff4444;
    }
    
    /* Chart Container */
    .chart-container {
        background-color: #1a1f2e;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #2d3748;
        margin: 10px 0;
    }
    
    .chart-title {
        color: #00ff88;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Results Table */
    .results-table {
        background-color: #1a1f2e;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #2d3748;
    }
    
    .results-row {
        display: flex;
        justify-content: space-between;
        padding: 12px 20px;
        border-bottom: 1px solid #2d3748;
    }
    
    .results-row:hover {
        background-color: #252c3d;
    }
    
    .result-method {
        color: #888;
        font-size: 13px;
        font-weight: 500;
    }
    
    .result-value {
        color: #00ff88;
        font-family: 'Courier New', monospace;
        font-weight: 600;
        font-size: 14px;
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #00ff88 0%, #00cc6f 100%);
        color: #000000;
        font-weight: 700;
        font-size: 16px;
        padding: 15px;
        border-radius: 8px;
        border: none;
        text-transform: uppercase;
        letter-spacing: 2px;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #00cc6f 0%, #00ff88 100%);
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1f2e;
        border-right: 2px solid #00ff88;
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Info/Success boxes */
    .stAlert {
        background-color: #1a1f2e;
        border: 1px solid #00ff88;
        border-radius: 8px;
    }
    
    /* Dataframe styling */
    .dataframe {
        background-color: #1a1f2e !important;
        color: #FAFAFA !important;
    }
    
    /* Input widgets */
    .stMultiSelect, .stSlider, .stNumberInput, .stRadio {
        background-color: #0E1117;
    }
    
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
def render_header(status="IDLE", backend_type="Simulator"):
    status_class = {
        "IDLE": "status-idle",
        "RUNNING": "status-running",
        "DONE": "status-done"
    }.get(status, "status-idle")
    
    st.markdown(f"""
    <div class="terminal-header">
        <span class="terminal-title">⚛️ OptiQ-Flow</span>
        <span class="status-badge {status_class}">STATUS: {status}</span>
        <span class="status-badge status-idle">BACKEND: {backend_type.upper()}</span>
    </div>
    """, unsafe_allow_html=True)

def render_metric_card(label, value, delta=None, is_positive=None):
    value_class = "metric-value"
    if delta is not None and is_positive is not None:
        if not is_positive:
            value_class += " negative"
    
    delta_html = ""
    if delta is not None:
        delta_class = "delta-positive" if is_positive else "delta-negative"
        delta_symbol = "▲" if is_positive else "▼"
        delta_html = f'<div class="metric-delta"><span class="{delta_class}">{delta_symbol} {abs(delta):.4f}</span></div>'
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="{value_class}">{value:.6f}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def create_convergence_chart(history):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(history))),
        y=history,
        mode='lines+markers',
        name='Energy',
        line=dict(color='#00ff88', width=2),
        marker=dict(size=6, color='#00ff88'),
        fill='tozeroy',
        fillcolor='rgba(0, 255, 136, 0.1)'
    ))
    
    fig.update_layout(
        title='QAOA Convergence',
        title_font=dict(size=14, color='#00ff88', family='Courier New'),
        xaxis_title='Iteration',
        yaxis_title='Energy',
        plot_bgcolor='#0E1117',
        paper_bgcolor='#1a1f2e',
        font=dict(color='#FAFAFA', family='Courier New'),
        xaxis=dict(
            gridcolor='#2d3748',
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            gridcolor='#2d3748',
            showgrid=True,
            zeroline=False
        ),
        hovermode='x unified',
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

def create_comparison_chart(c_val, raw, rem, zne):
    methods = ['Classical', 'Raw QAOA', 'Readout Mit.', 'ZNE Mit.']
    values = [c_val, raw, rem, zne]
    colors = ['#00ff88', '#ff4444', '#ff9500', '#00ccff']
    
    fig = go.Figure(data=[go.Bar(
        x=methods,
        y=values,
        marker_color=colors,
        text=[f'{v:.6f}' for v in values],
        textposition='outside',
        textfont=dict(color='#FAFAFA', family='Courier New', size=11)
    )])
    
    fig.update_layout(
        title='Energy Comparison',
        title_font=dict(size=14, color='#00ff88', family='Courier New'),
        yaxis_title='Energy',
        plot_bgcolor='#0E1117',
        paper_bgcolor='#1a1f2e',
        font=dict(color='#FAFAFA', family='Courier New'),
        xaxis=dict(
            gridcolor='#2d3748',
            showgrid=False
        ),
        yaxis=dict(
            gridcolor='#2d3748',
            showgrid=True,
            zeroline=False
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=False
    )
    
    return fig

# ==========================================
# MAIN APP
# ==========================================
def main():
    st.set_page_config(
        page_title="OptiQ-Flow | Quantum Portfolio Optimization",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    apply_trading_terminal_style()
    
    # Session state initialization
    if 'status' not in st.session_state:
        st.session_state.status = "IDLE"
    if 'results' not in st.session_state:
        st.session_state.results = None
    
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
    
    # Header
    backend_display = "SIMULATOR" if backend_type == "Simulator" else "IBM QUANTUM"
    render_header(st.session_state.status, backend_display)
    
    # Main execution
    if run_solver and tickers:
        st.session_state.status = "RUNNING"
        render_header(st.session_state.status, backend_display)
        
        progress_container = st.empty()
        
        # Step 1: Data Fetch
        with progress_container.container():
            with st.spinner("📊 Fetching market data..."):
                df = fetch_stock_data(tickers, config.START_DATE, config.END_DATE)
                mu, sigma = calculate_covariance_matrix(df)
        
        # Step 2: Classical Solver
        with progress_container.container():
            with st.spinner("⚙️ Running classical optimization..."):
                c_solver = ClassicalPortfolioSolver(mu, sigma, risk, budget)
                c_res = c_solver.solve()
        
        # Step 3: Quantum Setup
        with progress_container.container():
            with st.spinner("🔬 Initializing quantum circuit..."):
                qubo = PortfolioOptimizationQUBO(mu, sigma, risk, budget)
                op, _ = qubo.to_ising()
                backend = get_backend(backend_type == "Real Hardware")
        
        # Step 4: QAOA Execution
        with progress_container.container():
            with st.spinner("⚛️ Executing QAOA optimization..."):
                runner = RawQAOARunner(op, backend)
                q_res = runner.run()
        
        # Step 5: Error Mitigation
        with progress_container.container():
            with st.spinner("🛡️ Applying error mitigation..."):
                mitigator = MitigatedExecutor(backend, op)
                m_res = mitigator.execute(q_res.circuit)
        
        progress_container.empty()
        
        # Store results
        st.session_state.results = {
            'classical': c_res,
            'qaoa': q_res,
            'mitigated': m_res,
            'tickers': tickers
        }
        st.session_state.status = "DONE"
        st.rerun()
    
    # Display Results
    if st.session_state.results:
        results = st.session_state.results
        c_res = results['classical']
        q_res = results['qaoa']
        m_res = results['mitigated']
        selected_tickers = results['tickers']
        
        # Main content area
        col_main, col_side = st.columns([7, 3])
        
        with col_main:
            # Convergence Chart
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">📈 QAOA Energy Convergence</div>', unsafe_allow_html=True)
            st.plotly_chart(create_convergence_chart(q_res.history), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Comparison Chart
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">⚖️ Method Comparison</div>', unsafe_allow_html=True)
            st.plotly_chart(
                create_comparison_chart(c_res.value, m_res['raw'], m_res['rem'], m_res['zne']),
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_side:
            # Energy Metrics
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">⚡ ENERGY METRICS</div>', unsafe_allow_html=True)
            
            render_metric_card("Classical Baseline", c_res.value)
            
            raw_improvement = c_res.value < m_res['raw']
            raw_delta = m_res['raw'] - c_res.value
            render_metric_card("Raw QAOA", m_res['raw'], raw_delta, not raw_improvement)
            
            rem_improvement = c_res.value < m_res['rem']
            rem_delta = m_res['rem'] - c_res.value
            render_metric_card("Readout Mitigated", m_res['rem'], rem_delta, not rem_improvement)
            
            zne_improvement = c_res.value < m_res['zne']
            zne_delta = m_res['zne'] - c_res.value
            render_metric_card("ZNE Mitigated", m_res['zne'], zne_delta, not zne_improvement)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Portfolio Allocation
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">📊 OPTIMAL PORTFOLIO</div>', unsafe_allow_html=True)
            
            selected_assets = [selected_tickers[i] for i in c_res.indices]
            portfolio_df = pd.DataFrame({
                'Asset': selected_assets,
                'Weight': [1/len(selected_assets)] * len(selected_assets)
            })
            
            st.dataframe(
                portfolio_df.style.format({'Weight': '{:.2%}'}),
                use_container_width=True,
                hide_index=True
            )
            
            st.markdown(f"""
            <div style="margin-top: 15px; padding: 10px; background-color: #0E1117; border-radius: 4px;">
                <div style="color: #888; font-size: 11px;">SHARPE RATIO</div>
                <div style="color: #00ff88; font-size: 20px; font-family: 'Courier New', monospace; font-weight: 700;">
                    {c_res.sharpe:.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Bottom Results Summary
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">📋 DETAILED RESULTS</div>', unsafe_allow_html=True)
        
        summary_df = pd.DataFrame({
            'Method': ['Classical', 'Raw QAOA', 'Readout Mitigation', 'ZNE Mitigation'],
            'Energy': [c_res.value, m_res['raw'], m_res['rem'], m_res['zne']],
            'Delta vs Classical': [0.0, m_res['raw'] - c_res.value, m_res['rem'] - c_res.value, m_res['zne'] - c_res.value],
            'Improvement': ['Baseline', 
                           '✓' if m_res['raw'] < c_res.value else '✗',
                           '✓' if m_res['rem'] < c_res.value else '✗',
                           '✓' if m_res['zne'] < c_res.value else '✗']
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
    
    elif st.session_state.status == "IDLE":
        # Welcome screen
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px;">
            <h1 style="color: #00ff88; font-size: 48px; font-family: 'Courier New', monospace;">
                ⚛️ OptiQ-Flow
            </h1>
            <p style="color: #888; font-size: 18px; margin-top: 20px;">
                Quantum-Enhanced Portfolio Optimization Platform
            </p>
            <p style="color: #FAFAFA; font-size: 14px; margin-top: 30px; max-width: 600px; margin-left: auto; margin-right: auto;">
                Configure your portfolio parameters in the sidebar and click <strong style="color: #00ff88;">EXECUTE SOLVER</strong> 
                to run quantum optimization with advanced error mitigation techniques.
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    

