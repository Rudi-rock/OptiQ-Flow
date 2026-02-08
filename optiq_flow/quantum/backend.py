from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError, pauli_error
from qiskit_ibm_runtime import QiskitRuntimeService
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_backend(name: str = "aer_simulator", use_real_backend: bool = False):
    """
    Returns a quantum backend.
    
    Args:
        name: Name of the backend (e.g., 'ibm_brisbane', 'aer_simulator')
        use_real_backend: If True, connects to IBM Quantum. otherwise returns local simulator.
    """
    if use_real_backend:
        try:
            service = QiskitRuntimeService()
            backend = service.backend(name)
            print(f"Using real backend: {backend.name}")
            return backend
        except Exception as e:
            print(f"Failed to connect to real backend: {e}. Falling back to simulator.")
            return get_noisy_simulator()
    else:
        return get_noisy_simulator()

def get_noisy_simulator():
    """
    Creates a simulator with a custom noise model to mimic NISQ hardware.
    Crucial for demonstrating error mitigation.
    """
    noise_model = NoiseModel()
    
    # 1. Depolarizing Error (Gate Fidelity)
    # 1-qubit gate error (0.1%)
    error_1q = depolarizing_error(0.001, 1)
    # 2-qubit gate error (1%) - CNOT is usually noisy
    error_2q = depolarizing_error(0.01, 2)
    
    noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3', 'rz', 'sx', 'x'])
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz', 'ecr'])
    
    # 2. Readout Error (Measurement Fidelity)
    # 2% probability of flipping 0->1 or 1->0
    p1_0 = 0.02
    p0_1 = 0.02
    readout_error = ReadoutError([[1 - p1_0, p1_0], [p0_1, 1 - p0_1]])
    
    noise_model.add_all_qubit_readout_error(readout_error)
    
    backend = AerSimulator(noise_model=noise_model)
    print("Using noisy AerSimulator.")
    return backend

if __name__ == "__main__":
    backend = get_backend()
    print(backend)
