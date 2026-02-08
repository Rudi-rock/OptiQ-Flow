from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from typing import Dict, Any, List
import numpy as np
import sys
import os

# Import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from quantum.error_mitigation.readout_mitigation import ReadoutMitigator
from quantum.error_mitigation.zne import ZNEMitigator, global_folding

class MitigatedQAOAExecutor:
    def __init__(self, backend, hamiltonian: SparsePauliOp):
        self.backend = backend
        self.hamiltonian = hamiltonian
        self.readout_mitigator = None
        
    def calibrate_readout(self, num_qubits: int):
        """
        Calibrates the readout mitigator.
        """
        self.readout_mitigator = ReadoutMitigator(self.backend, num_qubits)
        self.readout_mitigator.calibrate()

    def execute(self, circuit: QuantumCircuit, zne_scales: List[float] = [1.0, 3.0, 5.0]) -> Dict[str, Any]:
        """
        Executes the circuit with various mitigation strategies.
        Returns a dictionary with results for:
        - raw
        - readout_mitigated
        - zne_mitigated (energy only)
        """
        num_qubits = circuit.num_qubits
        if self.readout_mitigator is None:
            self.calibrate_readout(num_qubits)
            
        results = {}
        
        # 1. Raw + Readout Mitigation Run (Scale=1)
        # We can reuse Scale=1 ZNE run for this if we are careful, 
        # but let's be explicit.
        
        # Prepare ZNE circuits
        # We need to compute energy for each scale.
        # Energy = <H> = sum (coeff * <P>)
        # <P> is computed from counts.
        
        zne_energies = []
        raw_energy = None
        rem_energy = None
        raw_counts = None
        rem_counts = None # Readout Error Mitigated Counts
        
        print(f"Running Mitigated QAOA with ZNE scales: {zne_scales}")
        
        for scale in zne_scales:
            scaled_qc = global_folding(circuit, scale)
            # Add measurement if missing
            if scaled_qc.num_clbits == 0:
                scaled_qc.measure_all()
                
            t_qc = transpile(scaled_qc, self.backend)
            job = self.backend.run(t_qc, shots=4096)
            counts = job.result().get_counts()
            
            # Apply Readout Mitigation to the counts
            mitigated_counts = self.readout_mitigator.apply_correction(counts)
            
            # Compute Energy from Mitigated Counts
            energy = self._compute_expectation(mitigated_counts)
            zne_energies.append(energy)
            
            if scale == 1.0:
                # Store base results
                raw_counts = counts
                rem_counts = mitigated_counts
                # Compute raw energy (no readout mitigation)
                raw_energy = self._compute_expectation(counts)
                rem_energy = energy
                
        # 2. Results
        zne_mitigator = ZNEMitigator(zne_scales)
        zne_value = zne_mitigator.extrapolate(zne_energies)
        
        return {
            "raw_value": raw_energy,
            "rem_value": rem_energy,
            "zne_value": zne_value,
            "raw_counts": raw_counts,
            "rem_counts": rem_counts,
            "zne_history": zne_energies
        }

    def _compute_expectation(self, counts: Dict[str, float]) -> float:
        """
        Computes <H> from counts (distribution).
        H = sum c_i P_i
        """
        total_energy = 0.0
        
        # Re-normalize counts just in case
        total_shots = sum(counts.values())
        
        for pauli, coeff in self.hamiltonian.label_iter():
            term_val = 0.0
            for bitstring, count in counts.items():
                prob = count / total_shots
                parity = 0
                # Calculate parity of bitstring with respect to Pauli Z terms
                # Pauli string e.g., "ZZI"
                # Bitstring e.g., "101"
                # Iterate and check Zs
                for i, char in enumerate(pauli):
                    if char == 'Z':
                        # Match with bitstring bit
                        # Qiskit bitstring is little endian? Start from right?
                        # SparsePauliOp labels are usually "ZZI" -> q2 q1 q0
                        # Bitstring "101" -> q2=1, q1=0, q0=1
                        # So indices match if we reverse one or just map correctly.
                        # Standard convention: label[i] corresponds to qubit[num_qubits - 1 - i]
                        # Let's verify.
                        # If label is "Z.." (q_N-1), bitstring starts with q_N-1.
                        # So indices match directly.
                        if bitstring[i] == '1':
                            parity += 1
                            
                meas_val = (-1)**parity
                term_val += meas_val * prob
                
            total_energy += float(np.real(coeff)) * term_val
            
        return total_energy

if __name__ == "__main__":
    from quantum.qaoa_raw import RawQAOARunner
    
    # Setup
    H = SparsePauliOp.from_list([("ZZ", 1.0)])
    backend = AerSimulator() # Need noise for this to mean anything?
    
    # Run Raw to get circuit
    # runner = RawQAOARunner(H, backend)
    # result = runner.run() # This optimizes.
    
    # Let's just create a dummy circuit
    qc = QuantumCircuit(2)
    qc.h([0,1])
    qc.cx(0,1)
    qc.rx(0.5, 0)
    
    executor = MitigatedQAOAExecutor(backend, H)
    res = executor.execute(qc)
    print("Results:", res)
