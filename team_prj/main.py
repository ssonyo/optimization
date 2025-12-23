import numpy as np
import matplotlib.pyplot as plt
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
    정박자(On-beat) 로직 기반 시뮬레이션:
    [입고 -> 판매 -> 폐기/노화] 순서로 수익과 데이터를 수집합니다.
    """
    product = Product(config)
    dm = DemandModel(config)
    env = InventoryEnv(config, product, dm)
    policy = FixedPolicy(config) if policy_type == "Fixed" else DynamicPolicy(config)
    
    # 통계 및 진단 플롯용 데이터 저장소
    ep_reward, ep_waste, ep_sales = 0, 0, 0
    mu_trace = []
    daily_profits = []
    bin_history = [] 
    demand_bin_trace = []
    
    # 사분할 진단 대시보드용 상세 이력
    history = {
        'orders': [], 'prices': [], 'demands': [], 
        'revenue': [], 'order_costs': [], 'holding_costs': [], 'waste_costs': []
    }

    for t in range(config.T):
        # 0. 상태 관측
        bin_history.append(product.inventory_bins.copy())

        # 1. 의사결정 (x_t, p_t)
        order_qty, prices = policy.get_action(product) 
        
        # 2. 환경 변화 (정박자 순서: 입고 후 판매 진행)
        result = env.step(order_qty, prices, policy_type=policy_type)
        
        # 3. 데이터 기록 (사분할 플롯용)
        history['orders'].append(order_qty)
        history['prices'].append(prices.copy())
        history['demands'].append(result['demand'])
        history['revenue'].append(result.get('revenue', 0))
        history['order_costs'].append(result.get('order_cost', 0))
        history['holding_costs'].append(result.get('holding_cost', 0))
        history['waste_costs'].append(result.get('waste', 0) * config.BASE_COST * config.WASTE_COST_FACTOR)

        # 4. 누적 통계 업데이트
        ep_reward += result['reward']
        ep_waste += result['waste']
        ep_sales += result['demand']
        daily_profits.append(result['reward'])
        demand_bin_trace.append(result['demand_per_bin'])
        
        # 5. 베이지안 업데이트 (Dynamic 전용)
        if policy_type == "Dynamic":
            policy.update_belief(result['demand'], prices, product)
            mu_trace.append(policy.mu_alpha)
        else:
            mu_trace.append(0.0) 
            
    return ep_reward, ep_waste, ep_sales, mu_trace, bin_history, daily_profits, demand_bin_trace, history

def plot_policy_diagnostic(fixed_history, dynamic_history, config):
    """
    Fixed와 Dynamic의 행동 및 비용 구조를 직접 비교하는 대시보드
    """
    T = len(fixed_history['orders'])
    days = np.arange(T)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 가격 전략 비교
    ax_p = axes[0, 0]
    dyn_prices = np.array(dynamic_history['prices'])
    ax_p.plot(days, dyn_prices[:, 0], label="Dynamic: Expiring (m=1)", color='red', alpha=0.6)
    ax_p.plot(days, dyn_prices[:, -1], label="Dynamic: Fresh (m=L)", color='darkgreen')
    ax_p.axhline(y=config.BASE_PRICE, color='black', linestyle='--', label="Fixed Price")
    ax_p.set_title("Price Strategy Comparison")
    ax_p.set_ylabel("Price ($)")
    ax_p.legend()

    # 2. 주문량 vs 실제 수요
    ax_o = axes[0, 1]
    ax_o.plot(days, dynamic_history['orders'], label="Dynamic Order", color='blue', alpha=0.7)
    ax_o.plot(days, dynamic_history['demands'], label="Actual Demand", color='gray', linestyle=':', alpha=0.5)
    ax_o.set_title("Order Quantity vs Demand (Dynamic)")
    ax_o.set_ylabel("Quantity")
    ax_o.legend()

    # 3. 수익/비용 구조 분해
    ax_c = axes[1, 0]
    labels = ['Revenue', 'Order Cost', 'Holding Cost', 'Waste Cost']
    def get_comp(h): return [np.sum(h['revenue']), -np.sum(h['order_costs']), 
                             -np.sum(h['holding_costs']), -np.sum(h['waste_costs'])]
    
    x = np.arange(len(labels))
    width = 0.35
    ax_c.bar(x - width/2, get_comp(fixed_history), width, label='Fixed', color='skyblue')
    ax_c.bar(x + width/2, get_comp(dynamic_history), width, label='Dynamic', color='orange')
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(labels)
    ax_c.set_title("Total Profit/Cost Breakdown")
    ax_c.axhline(0, color='black', lw=1)
    ax_c.legend()

    # 4. 누적 수익 추이 비교
    ax_r = axes[1, 1]
    f_net = np.cumsum(np.array(fixed_history['revenue']) - np.array(fixed_history['order_costs']) - np.array(fixed_history['waste_costs']))
    d_net = np.cumsum(np.array(dynamic_history['revenue']) - np.array(dynamic_history['order_costs']) - np.array(dynamic_history['waste_costs']))
    ax_r.plot(days, f_net, label="Fixed Net Profit", color='skyblue')
    ax_r.plot(days, d_net, label="Dynamic Net Profit", color='orange')
    ax_r.set_title("Cumulative Net Profit Growth")
    ax_r.set_ylabel("Net Profit ($)")
    ax_r.legend()

    plt.tight_layout()
    plt.show()

def main():
    config = SimConfig()
    collector = StatsCollector()
    dynamic_mu_histories, dynamic_daily_profits = [], []
    last_f_hist, last_d_hist = None, None
    last_fixed_bins, last_dynamic_bins = None, None

    print(f"--- Simulation Start: N={config.N}, T={config.T} ---")

    for i in range(config.N):
        r_f, w_f, s_f, _, bin_f, _, _, hist_f = run_single_episode(config, "Fixed")
        r_d, w_d, s_d, mu_t, bin_d, d_p, _, hist_d = run_single_episode(config, "Dynamic")
        
        collector.add_episode_result("Fixed", r_f, w_f, s_f)
        collector.add_episode_result("Dynamic", r_d, w_d, s_d)
        dynamic_mu_histories.append(mu_t)
        dynamic_daily_profits.append(d_p)

        if i == config.N - 1:
            last_fixed_bins, last_dynamic_bins = bin_f, bin_d
            last_f_hist, last_d_hist = hist_f, hist_d

        if (i + 1) % 100 == 0:
            print(f"Episode {i+1}/{config.N} completed...")

    if config.SHOW_DISTRIBUTION: plot_simulation_results(collector)
    if config.SHOW_LEARNING_CURVE:
        plot_learning_curve(np.array(dynamic_mu_histories), config.ALPHA_TRUE, np.array(dynamic_daily_profits))
    
    if last_fixed_bins and last_dynamic_bins:
        plot_inventory_aging(last_fixed_bins, title="Inventory Aging: Fixed Policy")
        plot_inventory_aging(last_dynamic_bins, title="Inventory Aging: Dynamic Policy")

    if last_f_hist and last_d_hist:
        plot_policy_diagnostic(last_f_hist, last_d_hist, config)

if __name__ == "__main__":
    main()