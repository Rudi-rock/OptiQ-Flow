import sys
import os
import argparse
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.global_config import config
from data.fetch_data import fetch_stock_data
from data.portfolio_data import calculate_covariance_matrix
from qubo.qubo_formulation import PortfolioOptimizationQUBO
from classical.classical_baseline import ClassicalPortfolioSolver
from quantum.backend import get_backend
from quantum.qaoa_raw import RawQAOARunner
from quantum.qaoa_mitigated import MitigatedQAOAExecutor
from qiskit import QuantumCircuit

def main():
    parser = argparse.ArgumentParser(description="OptiQ-Flow: Quantum Portfolio Optimization")
    parser.add_argument("--assets", nargs="+", default=["AAPL", "MSFT", "GOOGL", "AMZN"], help="List of ticker symbols")
    parser.add_argument("--budget", type=int, default=2, help="Number of assets to select")
    parser.add_argument("--backend", type=str, default="aer", help="Backend to use (aer or real)")
    args = parser.parse_args()

    print("\n" + "="*50)
    print("      OptiQ-Flow: Quantum Portfolio Optimization      ")
    print("="*50 + "\n")

    # 1. Data Layer
    print("[1] Fetching Financial Data...")
    df = fetch_stock_data(args.assets, config.START_DATE, config.END_DATE)
    mu, sigma = calculate_covariance_matrix(df)
    print(f"    Assets: {args.assets}")
    print(f"    Expected Returns (mu): {mu}")
    
    # 2. QUBO Formulation
    print("\n[2] Formulating QUBO...")
    qubo_solver = PortfolioOptimizationQUBO(mu, sigma, risk_aversion=config.RISK_AVERSION, budget=args.budget)
    op, offset = qubo_solver.to_ising()
    print(f"    Hamiltonian Terms: {len(op)}")
    print(f"    Offset: {offset}")
    
    # 3. Classical Baseline
    print("\n[3] Running Classical Baseline...")
    classical_solver = ClassicalPortfolioSolver(mu, sigma, risk_aversion=config.RISK_AVERSION, budget=args.budget)
    classical_res = classical_solver.solve()
    print(f"    Optimal Bitstring: {classical_res.bitstring}")
    print(f"    Optimal Value: {classical_res.value:.4f}")
    
    # 4. Quantum Layer
    print("\n[4] Running Quantum Pipelines...")
    backend = get_backend(use_real_backend=(args.backend == "real"))
    
    # 4a. Raw QAOA Optimization (Finding Params)
    print("    [4a] Optimizing QAOA Parameters (Raw)...")
    # Using RawQAOARunner to find parameters
    raw_runner = RawQAOARunner(op, backend, reps=config.QAOA_DEPTH)
    raw_res = raw_runner.run(maxiter=30) # Short run for demo
    print(f"         Found Params: {raw_res.optimal_params}")
    print(f"         Raw Energy (Optimization): {raw_res.optimal_value:.4f}")
    
    # 4b. Mitigated Execution
    print("    [4b] Executing Mitigated Pipeline...")
    # Use the circuit with optimal parameters from 4a
    optimal_circuit = raw_res.circuit
    
    mitigated_executor = MitigatedQAOAExecutor(backend, op)
    mitigated_res = mitigated_executor.execute(optimal_circuit, zne_scales=config.ZNE_SCALE_FACTORS)
    
    print("\n" + "-"*50)
    print("RESULTS SUMMARY")
    print("-"*50)
    print(f"Classical Baseline:   {classical_res.value:.4f}")
    print(f"Raw QAOA Value:       {mitigated_res['raw_value']:.4f}")
    print(f"Readout Mitigated:    {mitigated_res['rem_value']:.4f}")
    print(f"ZNE Mitigated:        {mitigated_res['zne_value']:.4f}")
    
    # Error calculation (assuming Classical is True Minimum)
    # Note: QAOA minimizes Hamiltonian. Classical value is also minimum.
    # Energy error = |E_q - E_c|
    err_raw = abs(mitigated_res['raw_value'] - classical_res.value)
    err_rem = abs(mitigated_res['rem_value'] - classical_res.value)
    err_zne = abs(mitigated_res['zne_value'] - classical_res.value)
    
    print("-" * 30)
    print(f"Error (Raw):          {err_raw:.4f}")
    print(f"Error (REM):          {err_rem:.4f}")
    print(f"Error (ZNE):          {err_zne:.4f}")
    print("-" * 30)
    
    if err_zne < err_raw:
        print("\n✅ SUCCESS: Mitigation improved accuracy!")
    else:
        print("\n⚠️ WARNING: Mitigation did not improve accuracy (Noise might be too high or method unstable).")

if __name__ == "__main__":
    main()
