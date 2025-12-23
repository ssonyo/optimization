import numpy as np

class InventoryEnv:
    def __init__(self, config, product, demand_model):
        self.config = config
        self.p = product
        self.dm = demand_model

    def step(self, order_qty, prices, policy_type):

        f_old = self.p.f_forecast
        sigma_d_sq_old = self.p.sigma_demand_sq

        current_bins = self.p.inventory_bins.copy()
        current_bins[-1] += order_qty 

        self.p.inventory_bins = current_bins 
        actual_bin_demands = self.dm.get_bin_actual_demands(self.p, prices, self.config.ALPHA_TRUE)
        expected_bin_demands = self.dm.get_bin_expected_demands(self.p, prices, self.config.ALPHA_TRUE)
        
        sales_per_bin = np.minimum(current_bins, actual_bin_demands)
        current_bins -= sales_per_bin # 판매된 만큼 재고 차감

        revenue = np.sum(sales_per_bin * prices)
        order_cost = order_qty * self.config.BASE_COST
        holding_cost = np.sum(current_bins) * self.config.HOLDING_COST

        waste_qty = current_bins[0] 
        waste_penalty = waste_qty * (self.config.BASE_COST * self.config.WASTE_COST_FACTOR)

        next_bins = np.zeros(self.config.SHELF_LIFE)
        next_bins[:-1] = current_bins[1:] 

        reward = revenue - order_cost - holding_cost - waste_penalty
        # print(revenue, order_cost, holding_cost, waste_penalty)

        eps_f = np.random.normal(0, self.config.SIGMA_F)
        f_new = max(1e-6, f_old + eps_f)
        
        theta = self.config.SMOOTHING_FACTOR
        total_actual = np.sum(actual_bin_demands)
        total_expected = np.sum(expected_bin_demands)
        new_sigma_d_sq = (1-theta)*sigma_d_sq_old + theta*(total_actual - total_expected)**2

        self.p.inventory_bins = next_bins
        self.p.f_forecast = f_new
        self.p.sigma_demand_sq = new_sigma_d_sq

        return {
            "reward": reward,
            "waste": waste_qty,         
            "revenue": revenue,          
            "order_cost": order_cost,     
            "holding_cost": holding_cost, 
            "demand": total_actual,
            "demand_per_bin": actual_bin_demands,
            "inventory_bins": next_bins.copy()
        }