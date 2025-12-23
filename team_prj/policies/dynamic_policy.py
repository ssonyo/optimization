import numpy as np
from scipy.stats import norm
from core.demand_model import DemandModel
from utils.bayesian import update_normal_posterior

class DynamicPolicy:
    def __init__(self, config):
        self.config = config
        self.mu_alpha = config.MU_ALPHA_0
        self.sigma_alpha = config.SIGMA_ALPHA_0

    def get_action(self, product_state):
        hat_alpha = norm.rvs(self.mu_alpha, self.sigma_alpha)
        hat_alpha = max(0.01, hat_alpha) 

        prices = self._determine_prices(hat_alpha, product_state)
        expected_d = DemandModel.calculate_expected_demand(product_state, prices, hat_alpha, self.config)
        
        sigma_t = np.sqrt(product_state.sigma_demand_sq)
        target_inv = expected_d + (self.config.THETA_SAFETY * sigma_t)
        order_qty = max(0, target_inv - np.sum(product_state.inventory_bins))
        
        return int(order_qty), prices

    def update_belief(self, observed_demand, prices, product_state):
        """
        Bayesian Update: 총합 수요(Observed)를 바탕으로 탄력성 추정
        """
        total_inv = np.sum(product_state.inventory_bins)
        if total_inv <= 0: return
        
        weights = product_state.inventory_bins / total_inv
        avg_price = np.sum(prices * weights)
        sigma_D = np.sqrt(product_state.sigma_demand_sq)
        expected_d = DemandModel.calculate_expected_demand(product_state, prices, self.mu_alpha, self.config)
        
        # kalman filter update
        self.mu_alpha, self.sigma_alpha = update_normal_posterior(
            mu_0 = self.mu_alpha,
            sigma_0 = self.sigma_alpha,
            observed_demand = observed_demand, # 실제 총 수요
            expected_demand = expected_d,      # 현재 추정치 기반 기대 총 수요
            sigma_D = sigma_D,
            avg_price = avg_price,
            base_price = self.config.BASE_PRICE
        )

    def _determine_prices(self, current_alpha, product_state):
        """
        policy price 1: based on alpha
        """
        prices = np.zeros(self.config.SHELF_LIFE)
        discount_factor = 0.1 / (current_alpha + 0.5)
        min_price = self.config.BASE_COST * 1.05 # 최소 마진 확보

        for m_idx in range(self.config.SHELF_LIFE):
            m = m_idx + 1
            # 남은 기한이 짧을수록(L-m이 클수록) 할인율 증가
            discount = (self.config.SHELF_LIFE - m) * discount_factor
            prices[m_idx] = max(min_price, self.config.BASE_PRICE * (1 - discount))
        return prices
    
    def _determine_prices2(self, current_alpha, product_state):
        """
        재고 기반 타겟 소진(Target-Clearing) 가격 결정
        """
        L = self.config.SHELF_LIFE
        prices = np.ones(L) * self.config.BASE_PRICE
        f_t = product_state.f_forecast
        min_p = self.config.BASE_COST * 1.1 # 최소 10% 마진 확보

        m_range = np.arange(1, L + 1)
        phi_bins = np.exp(-self.config.GAMMA_FRESHNESS * (L - m_range))

        for i in range(L):
            R_m = product_state.inventory_bins[i] # 해당 칸의 현재 재고
            natural_d = max((f_t / L) * phi_bins[i], 1e-6)
            
            if R_m > natural_d:
                price_drop = np.log(R_m / natural_d) / max(current_alpha, 0.05)
                target_p = self.config.BASE_PRICE - price_drop
                
                prices[i] = np.clip(target_p, min_p, self.config.BASE_PRICE)
            else:
                prices[i] = self.config.BASE_PRICE
                
        return prices

    def _determine_prices3(self, current_alpha, product_state):
        L = self.config.SHELF_LIFE
        prices = np.ones(L) * self.config.BASE_PRICE
        f_t = product_state.f_forecast
        min_p = self.config.BASE_COST * 1.1

        phi_bins = np.exp(-self.config.GAMMA_FRESHNESS * (L - np.arange(1, L + 1)))

        for i in range(L):
            R_m = product_state.inventory_bins[i]
            # 선제적 할인: 예상 수요의 85%만 되어도 대응 시작
            target_d = max((f_t / L) * phi_bins[i], 1e-6) * 0.85
            
            if R_m > target_d:
                price_drop = np.log(R_m / target_d) / max(current_alpha, 0.1)
                prices[i] = np.clip(self.config.BASE_PRICE - price_drop, min_p, self.config.BASE_PRICE)
            
            # 노이즈: 학습이 멈추지 않도록 모든 가격에 미세한 변동 추가
            exploration_noise = np.random.uniform(-1.0, 1.0) 
            prices[i] = np.clip(prices[i] + exploration_noise, min_p, self.config.BASE_PRICE)
                
        return prices