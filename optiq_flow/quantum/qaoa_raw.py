from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import BackendEstimator
from qiskit_aer import AerSimulator
import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Tuple, List, Dict

@dataclass
class QAOAResult:
    optimal_params: List[float]
    optimal_value: float
    history: List[float]
    circuit: QuantumCircuit
    distribution: Dict[str, float]

class RawQAOARunner:
    def __init__(self, hamiltonian: SparsePauliOp, backend, reps: int = 1):
        self.hamiltonian = hamiltonian
        self.backend = backend
        self.reps = reps
        self.ansatz = QAOAAnsatz(cost_operator=hamiltonian, reps=reps)
        # Using BackendEstimator for direct backend access (easier for noise models)
        self.estimator = BackendEstimator(backend=self.backend)
        self.history = []

    def _cost_func(self, params):
        """
        Evaluates the energy of the Hamiltonian for given parameters.
        """
        job = self.estimator.run(self.ansatz, self.hamiltonian, parameter_values=params)
        result = job.result()
        energy = result.values[0]
        self.history.append(energy)
        print(f"Eval: {len(self.history)}, Energy: {energy:.4f}, Params: {params}")
        return energy

    def run(self, maxiter: int = 50) -> QAOAResult:
        """
        Runs the QAOA optimization loop.
        """
        print(f"Starting Raw QAOA (p={self.reps}) on {self.backend.name}...")
        self.history = []
        
        # Initial guess (random or near 0)
        initial_point = np.random.uniform(-np.pi, np.pi, 2 * self.reps)
        
        # Optimization
        result = minimize(
            self._cost_func, 
            initial_point, 
            method='COBYLA', 
            options={'maxiter': maxiter}
        )
        
        optimal_params = result.x
        optimal_value = result.fun
        
        # Get final distribution
        bound_circuit = self.ansatz.assign_parameters(optimal_params)
        bound_circuit.measure_all()
        # Transpile for backend
        from qiskit import transpile
        t_qc = transpile(bound_circuit, self.backend)
        job = self.backend.run(t_qc, shots=1024)
        counts = job.result().get_counts()
        
        # Normalize counts
        total_shots = sum(counts.values())
        distribution = {k: v / total_shots for k, v in counts.items()}
        
        return QAOAResult(
            optimal_params=optimal_params.tolist(),
            optimal_value=optimal_value,
            history=self.history,
            circuit=bound_circuit,
            distribution=distribution
        )

if __name__ == "__main__":
    # Test
    # H = Z_0 Z_1
    H = SparsePauliOp.from_list([("ZZ", 1.0)])
    backend = AerSimulator()
    runner = RawQAOARunner(H, backend, reps=1)
    result = runner.run(maxiter=20)
    print(f"Optimal Value: {result.optimal_value}")
    print(f"Optimal Params: {result.optimal_params}")
