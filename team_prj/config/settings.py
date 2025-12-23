from dataclasses import dataclass, field
import math

@dataclass
class SimConfig:
    # 1. Simulation Execution Control
    T: int = 100                # Simulation duration (days)
    N: int = 1000               # Number of Monte Carlo iterations
    DEBUG_MODE: bool = False     # Whether to output detailed logs

    # 2. Economic Parameters (Components of Reward Function $C_t$)
    BASE_PRICE: float = 10.0     # Reference price ($\bar{p}$)
    BASE_COST: float = 5.0       # Unit cost ($c$)
    GAMMA_FRESHNESS: float = 0.4 # Freshness decay coefficient ($\gamma$)
    WASTE_COST_FACTOR: float = 0.8 # Waste cost factor ($w$): penalty of 3x unit cost
    HOLDING_COST: float = 0.2    # Holding cost ($h$)
    BAYESIAN_MODE: str = 'freshness' 

    # 3. Forecasting & Initial States
    SMOOTHING_FACTOR: float = 0.1   
    INITIAL_F_FORECAST: float = 100.0
    INITIAL_SIGMA_D_SQ: float = 225.0 
    INITIAL_SIGMA_F_SQ: float = 1.0
    SIGMA_F: float = math.sqrt(INITIAL_SIGMA_F_SQ)  # Standard deviation of forecast noise ($\epsilon^F$)

    # 4. Ground Truth Market Environment (Values unknown to the model) 
    ALPHA_TRUE: float = 0.5         # Actual price elasticity ($\alpha$)
    SIGMA_D_ACTUAL: float = 5.0    # Standard deviation of actual demand noise ($\epsilon^D$)
    THETA_SAFETY: float = 0.5       # Safety stock factor (z-score)
    SHELF_LIFE: int = 3             # Shelf life ($L$): kept at 3 days for "hard mode" setup

    # 5. Bayesian Elasticity Estimation (Belief $B_t$) 
    # Initial distribution of the elasticity estimated by the model (Normal Distribution)
    MU_ALPHA_0: float = 0.8         # Initial mean estimate (starts at 0.5 to induce convergence to 0.2)
    SIGMA_ALPHA_0: float = 0.2      # Initial uncertainty (higher values lead to faster learning)
    LEARNING_RATE: float = 0.05        # Learning rate for Bayesian updates

    # 6. Sensitivity Analysis Experimental Range 
    # Heatmap X-axis: testing various price elasticity environments
    SENSITIVITY_ALPHAS: list = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5])
    # Heatmap Y-axis: testing various shelf life environments
    SENSITIVITY_LIFES: list = field(default_factory=lambda: [2, 3, 5, 7, 10])

    # 7. Visualization Control (Visualization Flags) 
    SHOW_DISTRIBUTION: bool = True   # Profit distribution & waste rate graphs
    SHOW_LEARNING_CURVE: bool = True # Bayesian elasticity convergence curve
    SHOW_SENSITIVITY: bool = False   # Sensitivity analysis heatmap