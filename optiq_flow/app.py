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
# 1. CONFIGURATION
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
# 2. DATA LAYER
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
# 3. QUBO FORMULATION
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
        # Convert constraints to penalty terms
        conv = QuadraticProgramToQubo()
        qubo = conv.convert(self.qp)
        return qubo.to_ising()

# ==========================================
# 4. CLASSICAL SOLVER
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
# 5. QUANTUM BACKEND
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
# 6. ERROR MITIGATION (ZNE + READOUT)
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
    
    # 1. Strip measurements to get unitary part
    unitary = QuantumCircuit(circuit.num_qubits)
    for instr in circuit.data:
        if instr.operation.name != "measure":
            unitary.append(instr.operation, instr.qubits, instr.clbits)
            
    # 2. Fold unitary
    folded = unitary.copy()
    inv = unitary.inverse()
    for _ in range(k):
        folded.barrier()
        folded = folded.compose(inv)
        folded.barrier()
        folded = folded.compose(unitary)
        
    # 3. Re-append measurements (easy way: measure_all)
    folded.measure_all()
    return folded

class ZNEMitigator:
    def __init__(self, scales): self.scales = scales
    def extrapolate(self, energies):
        # Use numpy polyfit (degree 1) instead of sklearn
        # y = mx + c. Value at x=0 is c.
        # polyfit returns [m, c] for deg=1
        coeffs = np.polyfit(self.scales, energies, 1)
        return coeffs[1]

# ==========================================
# 7. QAOA EXECUTOR
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
        # Use AerEstimator which takes run_options (shots, seed, etc.)
        # and automatically uses the backend associated with it (or we pass it)
        # For AerEstimator, we pass backend_options or run_options
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
# 8. UI HELPERS
# ==========================================
def plot_results(c_val, raw, rem, zne):
    fig = go.Figure(data=[go.Bar(
        x=['Classical', 'Raw QAOA', 'Readout Mit.', 'ZNE Mit.'],
        y=[c_val, raw, rem, zne],
        marker_color=['green', 'red', 'orange', 'blue']
    )])
    fig.update_layout(title="Optimization Results (Energy)", yaxis_title="Energy")
    return fig

# ==========================================
# 9. MAIN APP
# ==========================================
def main():
    st.set_page_config(page_title="OptiQ-Flow", layout="wide")
    st.title("OptiQ-Flow: All-in-One Quantum Portfolio Solver")
    
    with st.sidebar:
        st.header("Settings")
        tickers = st.multiselect("Assets", ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"], default=["AAPL", "MSFT", "GOOGL", "AMZN"])
        budget = st.number_input("Budget", 1, len(tickers), 2)
        risk = st.slider("Risk Aversion", 0.0, 1.0, 0.5)
        backend_type = st.radio("Backend", ["Simulator", "Real HW"])
        run = st.button("RUN SOLVER")

    if run:
        st.info("Fetching Data...")
        df = fetch_stock_data(tickers, config.START_DATE, config.END_DATE)
        mu, sigma = calculate_covariance_matrix(df)
        
        st.info("Solving Classical Baseline...")
        c_solver = ClassicalPortfolioSolver(mu, sigma, risk, budget)
        c_res = c_solver.solve()
        
        st.info("Running Quantum Optimization...")
        qubo = PortfolioOptimizationQUBO(mu, sigma, risk, budget)
        op, _ = qubo.to_ising()
        
        backend = get_backend(backend_type == "Real HW")
        
        # QAOA
        runner = RawQAOARunner(op, backend)
        q_res = runner.run()
        
        # Mitigation
        mitigator = MitigatedExecutor(backend, op)
        m_res = mitigator.execute(q_res.circuit)
        
        # Results
        st.success("Done!")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Classical", f"{c_res.value:.4f}")
            st.metric("Raw QAOA", f"{m_res['raw']:.4f}")
        with col2:
            st.metric("ZNE Mitigated", f"{m_res['zne']:.4f}")
            
        st.plotly_chart(plot_results(c_res.value, m_res['raw'], m_res['rem'], m_res['zne']))
        st.line_chart(q_res.history)

if __name__ == "__main__":
    main()
