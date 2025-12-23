import numpy as np

def update_normal_posterior(mu_0, sigma_0, observed_demand, expected_demand, 
                            sigma_D, avg_price, base_price, learning_rate=0.01):
    delta_p = avg_price - base_price
    if abs(delta_p) < 1e-4: 
        return mu_0, sigma_0

    # 1. Prior 분산 및 안전 로그 변환
    prior_var = sigma_0**2 + (learning_rate**2) 
    safe_obs = np.clip(observed_demand, 1e-6, None)
    safe_exp = np.clip(expected_demand, 1e-6, None)
    
    # 2. Innovation
    y = np.log(safe_obs) - np.log(safe_exp)
    H = -delta_p  # d(ln D)/d(alpha) = -delta_p
    
    # 3. kalman gain
    observation_var = np.clip((sigma_D / safe_obs)**2, 1e-6, 1.0)
    S = (H**2 * prior_var) + observation_var
    K = (prior_var * H) / (S + 1e-9)
    
    # 4. 업데이트 및 Joseph Form 분산 보호
    mu_1 = mu_0 + K * y
    I_KH = 1.0 - K * H
    posterior_var = (I_KH**2 * prior_var) + (K**2 * observation_var)
    
    return np.clip(mu_1, 0.05, 2.0), np.clip(np.sqrt(max(1e-10, posterior_var)), 0.01, 1.0)