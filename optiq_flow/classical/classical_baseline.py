import numpy as np
from itertools import combinations
from dataclasses import dataclass
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.portfolio_data import calculate_covariance_matrix

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
        """
        Solves the portfolio optimization problem using brute force.
        Since N is small (4-6), this is extremely fast and guaranteed to be correct.
        """
        indices = list(range(self.num_assets))
        best_value = float('inf')
        best_indices = []
        
        # Iterate over all combinations of length 'budget'
        for combo in combinations(indices, self.budget):
            # Create weight vector (all 1.0 for selected assets, 0.0 otherwise)
            # Note: In this binary formulation, we assume equal weights for pure selection
            # But the objective function handles the terms correctly.
            
            x = np.zeros(self.num_assets)
            x[list(combo)] = 1.0
            
            # Objective: q * x.T * Sigma * x - (1-q) * mu.T * x
            risk = x.T @ self.sigma @ x
            return_val = self.mu.T @ x
            
            # Minimize this value
            obj_value = self.q * risk - (1 - self.q) * return_val
            
            if obj_value < best_value:
                best_value = obj_value
                best_indices = list(combo)
        
        # Construct result
        best_x = np.zeros(self.num_assets)
        best_x[best_indices] = 1.0
        
        final_risk = np.sqrt(best_x.T @ self.sigma @ best_x)
        final_return = self.mu.T @ best_x
        sharpe = final_return / final_risk if final_risk > 1e-6 else 0.0
        
        bitstring = "".join(["1" if i in best_indices else "0" for i in range(self.num_assets)])
        # Qiskit uses little-endian (right to left), but standard string is left to right.
        # Let's keep standard string order: index 0 is left-most char.
        
        return ClassicalResult(
            indices=best_indices,
            bitstring=bitstring,
            value=best_value,
            ret=final_return,
            volatility=final_risk,
            sharpe=sharpe
        )

if __name__ == "__main__":
    # Test
    mu = np.array([0.1, 0.2, 0.15, 0.05])
    sigma = np.array([
        [0.04, 0.01, 0.02, 0.0],
        [0.01, 0.05, 0.03, 0.01],
        [0.02, 0.03, 0.06, 0.0],
        [0.0, 0.01, 0.0, 0.02]
    ])
    solver = ClassicalPortfolioSolver(mu, sigma, risk_aversion=0.5, budget=2)
    result = solver.solve()
    print(f"Optimal Portfolio Indices: {result.indices}")
    print(f"Bitstring: {result.bitstring}")
    print(f"Objective Value: {result.value:.4f}")
    print(f"Sharpe Ratio: {result.sharpe:.4f}")
