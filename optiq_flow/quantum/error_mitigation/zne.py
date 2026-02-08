from qiskit import QuantumCircuit
import numpy as np
from typing import List, Tuple
from sklearn.linear_model import LinearRegression

def global_folding(circuit: QuantumCircuit, scale_factor: float) -> QuantumCircuit:
    """
    Applies global circuit folding to implementation scaling of noise.
    Scale factor must be an odd integer (1, 3, 5, ...).
    
    Relation: Real Scale R = 1 + 2k
    If user passes float, we round to nearest R.
    """
    # Round to nearest odd integer >= 1
    # scale = 1 -> k=0
    # scale = 3 -> k=1
    # scale = 5 -> k=2
    
    target_scale = max(1, scale_factor)
    # Find nearest odd integer
    # if scale is 2.9, nearest odd is 3.
    # if scale is 1.1, nearest odd is 1.
    
    k = int(round((target_scale - 1) / 2))
    k = max(0, k)
    
    if k == 0:
        return circuit.copy()
    
    # Global folding: U -> U (U_inv U)^k
    # Note: Qiskit's inverse() is symbolic.
    
    folded_qc = circuit.copy()
    inverse_qc = circuit.inverse()
    
    for _ in range(k):
        # Determine barrier usage later. For now just append.
        # Ideally we put barriers to prevent optimization
        folded_qc.barrier()
        folded_qc = folded_qc.compose(inverse_qc)
        folded_qc.barrier()
        folded_qc = folded_qc.compose(circuit)
        
    return folded_qc

class ZNEMitigator:
    def __init__(self, scale_factors: List[float] = [1, 3, 5]):
        self.scale_factors = scale_factors
        
    def extrapolate(self, energies: List[float]) -> float:
        """
        Extrapolate to scale factor 0 using Linear Regression (Richardson Extrapolation for 2 points, fits for >2).
        """
        if len(energies) != len(self.scale_factors):
            raise ValueError("Energies length must match scale factors")
            
        X = np.array(self.scale_factors).reshape(-1, 1)
        y = np.array(energies)
        
        # Fit linear model (noise vs scale)
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict at scale = 0
        zero_noise_val = model.predict([[0.0]])[0]
        return zero_noise_val

if __name__ == "__main__":
    # Test folding
    qc = QuantumCircuit(1)
    qc.h(0)
    
    print("Scale 1:")
    print(global_folding(qc, 1))
    
    print("\nScale 3:")
    print(global_folding(qc, 3))
    
    # Test extrapolation
    # Assume noise makes energy higher (minimize problem)
    scales = [1, 3, 5]
    energies = [-0.8, -0.7, -0.6] # Linear trend towards 0 error (-1.0 ideal)
    
    zne = ZNEMitigator(scales)
    result = zne.extrapolate(energies)
    print(f"\nExtrapolated (Scale 0): {result:.4f}")
    # Should be close to -0.85
