import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.result import Counts
from typing import List, Dict, Tuple
import scipy.linalg as la

class ReadoutMitigator:
    def __init__(self, backend, num_qubits: int):
        self.backend = backend
        self.num_qubits = num_qubits
        self.calibration_matrices = [] # List of 2x2 matrices for each qubit
        
    def calibrate(self):
        """
        Runs calibration circuits (all proposed qubits in 0 and all in 1)
        to characterize single-qubit readout errors.
        Assumes uncorrelated errors (tensored approach).
        """
        print(f"Calibrating Readout Error for {self.num_qubits} qubits...")
        
        # Circuit 0: All |0>
        qc0 = QuantumCircuit(self.num_qubits, self.num_qubits)
        qc0.measure(range(self.num_qubits), range(self.num_qubits))
        
        # Circuit 1: All |1>
        qc1 = QuantumCircuit(self.num_qubits, self.num_qubits)
        qc1.x(range(self.num_qubits))
        qc1.measure(range(self.num_qubits), range(self.num_qubits))
        
        circuits = [qc0, qc1]
        t_circuits = transpile(circuits, self.backend)
        job = self.backend.run(t_circuits, shots=2048)
        result = job.result()
        
        counts0 = result.get_counts(0)
        counts1 = result.get_counts(1)
        
        self.calibration_matrices = []
        
        for i in range(self.num_qubits):
            # Compute marginal probabilities for qubit i
            # P(measured=0 | prepared=0)
            p0_given_0 = self._get_marginal_prob(counts0, i, '0')
            # P(measured=1 | prepared=0) = 1 - p0_given_0
            
            # P(measured=1 | prepared=1)
            p1_given_1 = self._get_marginal_prob(counts1, i, '1')
            # P(measured=0 | prepared=1) = 1 - p1_given_1
            
            # Construct confusion matrix M
            # M = [[P(0|0), P(0|1)], 
            #      [P(1|0), P(1|1)]]
            # ideally [[1, 0], [0, 1]]
            
            M = np.array([
                [p0_given_0, 1 - p1_given_1],
                [1 - p0_given_0, p1_given_1]
            ])
            
            self.calibration_matrices.append(M)
            
        print("Calibration complete.")

    def _get_marginal_prob(self, counts: Dict[str, int], qubit_index: int, target_state: str) -> float:
        """
        Calculates marginal probability of a qubit being in target_state.
        Note: Qiskit counts keys are little-endian (qubit 0 is rightmost).
        """
        total = sum(counts.values())
        match_count = 0
        
        for bitstring, count in counts.items():
            # qubit_index 0 is the last character in bitstring
            # bitstring is length num_qubits
            # char index = num_qubits - 1 - qubit_index
            char_idx = self.num_qubits - 1 - qubit_index
            if bitstring[char_idx] == target_state:
                match_count += count
                
        return match_count / total

    def apply_correction(self, raw_counts: Dict[str, int]) -> Dict[str, float]:
        """
        Applies the inverse of the calibration matrices to the raw counts.
        Uses a simplified approach: probability vector correction.
        """
        if not self.calibration_matrices:
           raise RuntimeError("ReadoutMitigator not calibrated. Run calibrate() first.")
           
        # Convert counts to probability vector
        # Size 2^N
        dim = 2**self.num_qubits
        vec = np.zeros(dim)
        total_shots = sum(raw_counts.values())
        
        for bitstring, count in raw_counts.items():
            # Convert bitstring to integer index
            idx = int(bitstring, 2)
            vec[idx] = count / total_shots
            
        # Build full tensor product of inverses
        # M_inv_total = M_inv_0 (x) M_inv_1 (x) ...
        # But constructing the full matrix is expensive (2^N x 2^N).
        # Better to apply sequentially or just build it since N is small (<=6).
        # For N=6, dim=64. 64x64 is tiny. We can build it.
        
        M_total_inv = np.array([[1.0]])
        
        # Loop over qubits (careful with ordering! Qiskit is little-endian)
        # Tensor product order: q_n-1 (x) ... (x) q_0
        for i in range(self.num_qubits):
            M = self.calibration_matrices[i]
            try:
                M_inv = la.inv(M)
            except la.LinAlgError:
                print(f"Warning: Singular matrix for qubit {i}. unexpected.")
                M_inv = np.eye(2)
                
            M_total_inv = np.kron(M_inv, M_total_inv)
            
        # Apply correction
        corrected_vec = M_total_inv @ vec
        
        # Post-processing: remove negatives and renormalize
        # (Inverse method can produce negative probabilities)
        # Nearest probability distribution
        corrected_vec = np.clip(corrected_vec, 0, 1)
        corrected_vec /= np.sum(corrected_vec)
        
        # Convert back to dict
        corrected_counts = {}
        for i in range(dim):
            if corrected_vec[i] > 1e-6: # Filter tiny values
                # Convert int to bitstring with padding
                bitstring = format(i, f'0{self.num_qubits}b')
                corrected_counts[bitstring] = corrected_vec[i]
                
        return corrected_counts

if __name__ == "__main__":
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError
    
    # Setup noise
    noise_model = NoiseModel()
    p1_0 = 0.05
    readout_error = ReadoutError([[1 - p1_0, p1_0], [p1_0, 1 - p1_0]])
    noise_model.add_all_qubit_readout_error(readout_error)
    backend = AerSimulator(noise_model=noise_model)
    
    mitigator = ReadoutMitigator(backend, num_qubits=2)
    mitigator.calibrate()
    
    # Test correction
    # Ideally should be 50-50 00 and 11
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    
    counts = backend.run(transpile(qc, backend), shots=5000).result().get_counts()
    print("Raw Counts:", counts)
    
    corrected = mitigator.apply_correction(counts)
    print("Corrected:", corrected)
