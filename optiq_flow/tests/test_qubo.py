import unittest
import numpy as np
from qiskit_optimization import QuadraticProgram
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qubo.qubo_formulation import PortfolioOptimizationQUBO

class TestPortfolioQUBO(unittest.TestCase):
    def setUp(self):
        self.mu = np.array([0.1, 0.2])
        self.sigma = np.array([[0.1, 0.0], [0.0, 0.1]])
        self.risk_aversion = 0.5
        self.budget = 1
        self.qubo = PortfolioOptimizationQUBO(self.mu, self.sigma, self.risk_aversion, self.budget)

    def test_qp_creation(self):
        qp = self.qubo.create_quadratic_program()
        self.assertIsInstance(qp, QuadraticProgram)
        self.assertEqual(qp.get_num_binary_vars(), 2)
        # Check budget constraint
        constraint = qp.get_linear_constraint("budget")
        self.assertEqual(constraint.rhs, 1)

    def test_ising_conversion(self):
        op, offset = self.qubo.to_ising()
        # With 2 qubits, operator should have Pauli terms
        self.assertEqual(op.num_qubits, 2)
        # Offset should be float
        self.assertIsInstance(offset, float)

if __name__ == '__main__':
    unittest.main()
