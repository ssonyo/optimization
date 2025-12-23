import numpy as np

class DemandModel:
    def __init__(self, config):
        self.config = config

    def get_bin_expected_demands(self, product, prices, alpha):
        f_t = product.f_forecast
        L = self.config.SHELF_LIFE
        m = np.arange(1, L + 1) 
        
        phi = np.exp(-self.config.GAMMA_FRESHNESS * (L - m))
        psi = np.exp(-alpha * (prices - self.config.BASE_PRICE))
        
        return (f_t / L) * phi * psi

    def get_bin_actual_demands(self, product, prices, alpha_true):
        expected_demands = self.get_bin_expected_demands(product, prices, alpha_true)
        
        bin_sigma = self.config.SIGMA_D_ACTUAL / np.sqrt(self.config.SHELF_LIFE)
        noises = np.random.normal(0, bin_sigma, self.config.SHELF_LIFE)
        actual_bin_demands = np.maximum(0, expected_demands + noises)
        
        return actual_bin_demands

    @staticmethod
    def calculate_expected_demand(product, prices, alpha, config):
        dm = DemandModel(config)
        bin_demands = dm.get_bin_expected_demands(product, prices, alpha)
        return np.sum(bin_demands)