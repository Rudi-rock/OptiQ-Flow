import os
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class GlobalConfig:
    # Financial Parameters
    ASSETS: List[str] = None # Will be set dynamically or defaults
    START_DATE: str = "2023-01-01"
    END_DATE: str = "2024-01-01"
    RISK_AVERSION: float = 0.5  # Lambda (0 = max return, 1 = min risk)
    BUDGET: int = 1 # Number of assets to select (B)
    
    # Quantum Parameters
    QAOA_DEPTH: int = 1
    SHOTS: int = 1024
    BACKEND_NAME: str = "aer_simulator" # or "ibm_brisbane", etc.
    USE_REAL_BACKEND: bool = False
    
    # Mitigation Parameters
    ENABLE_MITIGATION: bool = True
    ZNE_SCALE_FACTORS: List[float] = None
    
    # Reproducibility
    SEED: int = 42

    def __post_init__(self):
        if self.ASSETS is None:
            self.ASSETS = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        if self.ZNE_SCALE_FACTORS is None:
            self.ZNE_SCALE_FACTORS = [1.0, 2.0, 3.0]

# Singleton instance
config = GlobalConfig()
