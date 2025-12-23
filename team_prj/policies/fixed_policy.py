import numpy as np
from .base_policy import BasePolicy
from core.demand_model import DemandModel

class FixedPolicy(BasePolicy):
    def __init__(self, config):
        super().__init__(config)

    def get_action(self, product):

        # set fixed prices
        prices = np.full(self.config.SHELF_LIFE, self.config.BASE_PRICE)
        
        # calculate expected demand
        expected_d = DemandModel.calculate_expected_demand(
            product, prices, alpha=0.0, config=self.config
        )

        # decide order quantity
        sigma_t = np.sqrt(product.sigma_demand_sq)
        total_inventory = np.sum(product.inventory_bins)

        target_inventory = expected_d + self.config.THETA_SAFETY * sigma_t
        order_qty = max(0, target_inventory - total_inventory)
        
        return int(order_qty), prices
    