
from AdaptiveMarketPlanningModel import AdaptiveMarketPlanningModel
import numpy as np

class ParametricModel(AdaptiveMarketPlanningModel):
    def __init__(self, state_names, x_names, s_0, T, reward_type,
                 cost=1.0, price_low=1.0, price_high=10.0,
                 price_process='RW', seed=20180613):
        super().__init__(state_names, x_names, s_0, T, reward_type,
                         cost=cost, seed=seed)
        self.low = float(price_low)
        self.high = float(price_high)
        self.PRICE_PROCESS = price_process

        # Kesten-style sign flip 체크용
        self.past_derivative = np.array([0.0, 0.0, 0.0], dtype=float)

    # order 및 gradient 함수
    def order_quantity_fn(self, price, theta):
        # q = max(0, theta0 + theta1 * p + theta2 * p^{-2})
        return max(0.0, float(theta[0] + theta[1] * price + theta[2] * price ** (-2)))

    def derivative_fn(self, price, theta):
        # dq/dtheta = [1, p, p^{-2}]
        return np.array([1.0, price, price ** (-2)], dtype=float)

    # 가격 프로세스
    def _draw_next_price(self, current_price, rng):
        if self.PRICE_PROCESS == 'RW':
            u = rng.uniform()
            delta = -1.0 if u < 0.2 else (1.0 if u > 0.8 else 0.0)
            p_next = current_price + delta
        else:
            p_next = rng.uniform(self.low, self.high)
        return float(min(self.high, max(self.low, p_next)))

    # exogenous info
    def exog_info_fn(self, decision):
        demand = float(self.prng.exponential(100.0))
        price_next = float(self._draw_next_price(self.state.price, self.prng))
        return {"demand": demand, "price_next": price_next}

    # 목적함수
    def objective_fn(self, decision, exog_info):
        q = self.order_quantity_fn(self.state.price, self.state.theta)
        self.order_quantity = q

        p_next = exog_info["price_next"]
        d = exog_info["demand"]
        # Ft = p_{t+1} * min(q, d) - c*q
        return p_next * min(q, d) - self.cost * q

    # SGD 업데이트 
    def transition_fn(self, decision, exog_info):
        self.learning_list.append(self.state.theta.copy())

        # gradient용 가격도 p_{t+1}
        p_for_grad = exog_info["price_next"]

        q = self.order_quantity_fn(self.state.price, self.state.theta)
        dq_dtheta = self.derivative_fn(self.state.price, self.state.theta)

        if q < exog_info["demand"]:
            grad = (p_for_grad - self.cost) * dq_dtheta
        else:
            grad = (-self.cost) * dq_dtheta

        # 부호 전환 카운팅
        new_counter = self.state.counter + 1 if np.dot(self.past_derivative, grad) < 0 else self.state.counter
        self.past_derivative = grad.copy()

        # theta, price업데이트
        new_theta = self.state.theta + decision.step_size * grad
        new_price = exog_info["price_next"]

        return {"counter": int(new_counter), "price": float(new_price), "theta": new_theta}
