# team_prj/core/product.py
import numpy as np

class Product:
    def __init__(self, config):
        self.config = config
        self.shelf_life = config.SHELF_LIFE
        self.base_price = config.BASE_PRICE
        self.cost = config.BASE_COST
        self.reset()

    def reset(self):
        """initialize product state variables on the new episode"""
        self.inventory_bins = np.array([40.0, 30.0, 0.0])
        self.f_forecast = getattr(self.config, 'INITIAL_F_FORECAST', 40.0)
        self.sigma_demand_sq = getattr(self.config, 'INITIAL_SIGMA_D_SQ', 225.0)
        self.sigma_f_sq = getattr(self.config, 'INITIAL_SIGMA_F_SQ', 1.0)