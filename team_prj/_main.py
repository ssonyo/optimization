import numpy as np
from config.settings import SimConfig
from core.product import Product
from core.demand_model import DemandModel
from core.base_env import InventoryEnv
from policies.fixed_policy import FixedPolicy
from policies.dynamic_policy import DynamicPolicy
from utils.stats_collector import StatsCollector
from visualization.plotter import (
    plot_simulation_results, 
    plot_learning_curve, 
    plot_sensitivity_heatmap,
    plot_inventory_aging
)

def run_single_episode(config, policy_type):
    """
    S_{t+1} = S^M(S_t, x_t, W_{t+1}) framework.
    Now tracks bin-level demand for advanced analysis.
    """
    product = Product(config)
    dm = DemandModel(config)
    env = InventoryEnv(config, product, dm)
    policy = FixedPolicy(config) if policy_type == "Fixed" else DynamicPolicy(config)
    
    # Cumulative Metrics
    ep_reward, ep_waste, ep_sales = 0, 0, 0
    
    # Time-series Traces
    mu_trace = []
    daily_profits = []
    bin_history = [] 
    demand_bin_trace = [] # [New] Track demand distribution across bins

    for t in range(config.T):
        # 0. State Observation
        bin_history.append(product.inventory_bins.copy())

        # 1. Action Decision (x_t)
        order_qty, prices = policy.get_action(product) 
        
        # 2. Transition (W_{t+1}, S_{t+1})
        # Note: result['demand'] is aggregated total demand
        result = env.step(order_qty, prices, policy_type=policy_type)
        
        # 3. Metrics Update
        ep_reward += result['reward']
        ep_waste += result['waste']
        ep_sales += result['demand'] # Aggregated demand used for total sales
        daily_profits.append(result['reward'])
        
        # 4. [New] Record Bin-level Demand for heatmap/analysis
        demand_bin_trace.append(result['demand_per_bin'])
        
        # 5. Belief Update (B_t)
        if policy_type == "Dynamic":
            # Bayesian update still uses total aggregated demand
            policy.update_belief(result['demand'], prices, product)
            mu_trace.append(policy.mu_alpha)
        else:
            mu_trace.append(0.0) 
            
    # Return all metrics, including the new bin-level demand trace
    return ep_reward, ep_waste, ep_sales, mu_trace, bin_history, daily_profits, demand_bin_trace

def main():
    config = SimConfig()
    collector = StatsCollector()
    
    dynamic_mu_histories = []
    dynamic_daily_profits = []
    # [New] For detailed bin-demand visualization across episodes
    dynamic_bin_demand_histories = [] 

    last_fixed_bins = None
    last_dynamic_bins = None

    print(f"--- Simulation Start: N={config.N}, T={config.T} ---")

    for i in range(config.N):
        # --- [1] Fixed Policy ---
        # Unpack the 7th value (bin_demand_trace) using '_' if not used immediately
        r_f, w_f, s_f, _, bin_f, _, _ = run_single_episode(config, "Fixed")
        collector.add_episode_result("Fixed", r_f, w_f, s_f)
        
        # --- [2] Dynamic Policy ---
        r_d, w_d, s_d, mu_t, bin_d, d_p, d_bin_trace = run_single_episode(config, "Dynamic")
        collector.add_episode_result("Dynamic", r_d, w_d, s_d)
        
        dynamic_mu_histories.append(mu_t)
        dynamic_daily_profits.append(d_p)
        dynamic_bin_demand_histories.append(d_bin_trace)

        if (i + 1) % 100 == 0:
            print(f"Episode {i+1}/{config.N} completed...")

        if i == config.N - 1:
            last_fixed_bins = bin_f
            last_dynamic_bins = bin_d

    # --- Visualizations ---
    if config.SHOW_DISTRIBUTION:
        plot_simulation_results(collector)
        
    if config.SHOW_LEARNING_CURVE:
        mu_arr = np.array(dynamic_mu_histories)
        profit_arr = np.array(dynamic_daily_profits)
        plot_learning_curve(mu_arr, config.ALPHA_TRUE, profit_arr)

    if last_fixed_bins and last_dynamic_bins:
        plot_inventory_aging(last_fixed_bins, title="Inventory Aging: Fixed Policy")
        plot_inventory_aging(last_dynamic_bins, title="Inventory Aging: Dynamic Policy")

    if config.SHOW_SENSITIVITY:
        run_sensitivity_analysis(config)

def run_sensitivity_analysis(config):
    print("\n--- Running Sensitivity Analysis Heatmap ---")
    heatmap_matrix = []
    for L in config.SENSITIVITY_LIFES:
        row = []
        for a in config.SENSITIVITY_ALPHAS:
            temp_config = SimConfig()
            temp_config.SHELF_LIFE = L
            temp_config.ALPHA_TRUE = a
            # Ensure proper unpacking of the 7 return values
            row.append(np.mean([run_single_episode(temp_config, "Dynamic")[0] for _ in range(30)]))
        heatmap_matrix.append(row)
    plot_sensitivity_heatmap(heatmap_matrix, config.SENSITIVITY_ALPHAS, config.SENSITIVITY_LIFES)

if __name__ == "__main__":
    main()