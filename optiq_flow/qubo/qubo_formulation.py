from qiskit_optimization import QuadraticProgram
from qiskit.quantum_info import SparsePauliOp
import numpy as np

class PortfolioOptimizationQUBO:
    def __init__(self, mu: np.ndarray, sigma: np.ndarray, risk_aversion: float, budget: int):
        """
        Initialize the QUBO formulator.
        
        Args:
            mu: Expected returns vector
            sigma: Covariance matrix
            risk_aversion: Lambda parameter (0=max return, 1=min risk)
            budget: Number of assets to select (B)
        """
        self.mu = mu
        self.sigma = sigma
        self.q = risk_aversion 
        self.budget = budget
        self.num_assets = len(mu)
        self.qp = None
        
    def create_quadratic_program(self) -> QuadraticProgram:
        """
        Constructs the Quadratic Program for Portfolio Optimization.
        Minimize: q * x.T * Sigma * x - (1-q) * mu.T * x
        Subject to: sum(x) == budget
        """
        qp = QuadraticProgram()
        
        # Add binary variables
        for i in range(self.num_assets):
            qp.binary_var(name=f"x_{i}")
            
        # Objective function
        # q * x^T * Sigma * x
        # Note: qiskit-optimization minimizes by default
        quadratic = {}
        for i in range(self.num_assets):
            for j in range(self.num_assets):
                quadratic[(f"x_{i}", f"x_{j}")] = self.q * self.sigma[i, j]
                
        # - (1-q) * mu^T * x
        linear = {}
        for i in range(self.num_assets):
            linear[f"x_{i}"] = -(1 - self.q) * self.mu[i]
            
        qp.minimize(linear=linear, quadratic=quadratic)
        
        # Constraint: sum(x) == budget
        linear_constraint = {f"x_{i}": 1 for i in range(self.num_assets)}
        qp.linear_constraint(linear=linear_constraint, sense="==", rhs=self.budget, name="budget")
        
        self.qp = qp
        return qp
    
    def to_ising(self) -> Tuple[SparsePauliOp, float]:
        """
        Converts the Quadratic Program to an Ising Hamiltonian.
        Penalty terms are automatically handled by from_ising.
        """
        if self.qp is None:
            self.create_quadratic_program()
            
        # Convert to Ising Hamiltonian
        # Note: We assume standard penalty handling for constraints
        # qiskit-optimization's to_ising maps x -> (I - Z)/2
        op, offset = self.qp.to_ising()
        return op, offset

if __name__ == "__main__":
    # Test run
    mu = np.array([0.1, 0.2, 0.15, 0.05])
    sigma = np.array([
        [0.04, 0.01, 0.02, 0.0],
        [0.01, 0.05, 0.03, 0.01],
        [0.02, 0.03, 0.06, 0.0],
        [0.0, 0.01, 0.0, 0.02]
    ])
    
    qubo_solver = PortfolioOptimizationQUBO(mu, sigma, risk_aversion=0.5, budget=2)
    qp = qubo_solver.create_quadratic_program()
    print("Quadratic Program:")
    print(qp.export_as_lp_string())
    
    op, offset = qubo_solver.to_ising()
    print("\nIsing Hamiltonian:")
    print(op)
