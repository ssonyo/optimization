import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_simulation_results(stats):
    """
    StatsCollector로부터 받은 데이터를 바탕으로 성과를 비교합니다.
    1. 정책별 누적 수익 분포 (Profit Distribution)
    2. 정책별 폐기율 비교 (Waste Rate Comparison)
    """
    fixed_rewards = [d['reward'] for d in stats.results['Fixed']]
    dynamic_rewards = [d['reward'] for d in stats.results['Dynamic']]
    
    fixed_waste = [d['waste_rate'] for d in stats.results['Fixed']]
    dynamic_waste = [d['waste_rate'] for d in stats.results['Dynamic']]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # [그래프 1] 수익 분포 (Histogram & KDE)
    sns.histplot(fixed_rewards, color="skyblue", label="Fixed (LIFO)", kde=True, ax=axes[0])
    sns.histplot(dynamic_rewards, color="orange", label="Dynamic (FIFO)", kde=True, ax=axes[0])
    axes[0].set_title("Cumulative Profit Distribution")
    axes[0].set_xlabel("Total Profit")
    axes[0].legend()

    # [그래프 2] 폐기율 비교 (Boxplot)
    sns.boxplot(data=[fixed_waste, dynamic_waste], palette=["skyblue", "orange"], ax=axes[1])
    axes[1].set_xticklabels(["Fixed (LIFO)", "Dynamic (FIFO)"])
    axes[1].set_title("Waste Rate Comparison")
    axes[1].set_ylabel("Waste Rate (%)")

    plt.tight_layout()
    plt.savefig("simulation_result.png")
    print("\n[The result plot is saved as 'simulation_result.png'")
    plt.show()


def plot_learning_curve(mu_history, true_alpha, daily_profit_history):
    """
    X축을 T(Days)로 통일하여 '지능의 성장'과 '수익의 증가'를 같은 시간선상에서 비교
    """
    # mu_history: (N, T) matrix
    # daily_profit_history: (N, T) matrix
    
    T = len(mu_history[0])  # 300일
    days = np.arange(T)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- 왼쪽: Alpha 수렴 ---
    avg_mu = np.mean(mu_history, axis=0)
    std_mu = np.std(mu_history, axis=0)

    ax1.plot(days, avg_mu, label="Estimated Alpha ($\mu_\\alpha$)", color='blue', lw=2)
    ax1.fill_between(days, avg_mu - std_mu, avg_mu + std_mu, color='blue', alpha=0.2)
    ax1.axhline(y=true_alpha, color='red', linestyle='--', label=f"True Alpha ({true_alpha})")
    ax1.set_title(f"1. Alpha Convergence (Over {T} Days)", fontsize=12)
    ax1.set_xlabel("Day (t)")
    ax1.set_ylabel("Elasticity ($\\alpha$)")
    ax1.set_xlim(0, T) # X축을 정확히 T까지 고정
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- 오른쪽: 일일 이익 추이 (성과) ---
    avg_profit = np.mean(daily_profit_history, axis=0)
    std_profit = np.std(daily_profit_history, axis=0)

    ax2.plot(days, avg_profit, label="Avg Daily Profit", color='green', lw=2)
    ax2.fill_between(days, avg_profit - std_profit, avg_profit + std_profit, color='green', alpha=0.1)
    
    # 이익 최적화 지점(Sweet Spot)을 보여주는 추세선
    z = np.polyfit(days, avg_profit, 3) 
    p = np.poly1d(z)
    ax2.plot(days, p(days), "darkgreen", linestyle="--", label="Profit Trend")

    ax2.set_title(f"2. Daily Profit Evolution (Over {T} Days)", fontsize=12)
    ax2.set_xlabel("Day (t)")
    ax2.set_ylabel("Daily Profit ($)")
    ax2.set_xlim(0, T) # X축을 정확히 T까지 고정
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_sensitivity_heatmap(data_matrix, alpha_range, shelf_life_range):
    """③ 감도 분석 Heatmap: 특정 환경에서의 수익성 확인"""
    plt.figure(figsize=(10, 8))
    
    df = pd.DataFrame(data_matrix, index=shelf_life_range, columns=alpha_range)
    sns.heatmap(df, annot=True, fmt=".1f", cmap="YlGnBu")
    
    plt.title("Sensitivity Analysis: Total Profit by Alpha & Shelf Life")
    plt.xlabel("Price Elasticity ($\\alpha$)")
    plt.ylabel("Shelf Life ($L$)")
    plt.show()


def plot_inventory_aging(bin_history, title="Inventory Composition by Remaining Shelf Life"):
    """
    시간에 따른 재고의 구성(남은 유통기한별)을 시각화합니다.
    bin_history: (T, L) 형태의 리스트 또는 배열
    """
    history = np.array(bin_history)
    T, L = history.shape
    days = np.arange(T)
    
    plt.figure(figsize=(12, 6))
    # 남은 유통기한이 짧은 순서(0일 남음, 1일 남음...)대로 쌓습니다.
    labels = [f"{i+1} Day(s) Left" for i in range(L)]
    
    plt.stackplot(days, history.T, labels=labels, alpha=0.8, edgecolors='white')
    
    plt.title(title, fontsize=14)
    plt.xlabel("Time Step (Day)", fontsize=12)
    plt.ylabel("Inventory Quantity", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_inventory_flow(history):
    """단일 에피소드 동안의 재고 흐름을 시각화 (선택 사항)"""
    plt.figure(figsize=(10, 5))
    plt.plot(history['inventory'], label='Inventory Level', color='green')
    plt.plot(history['demand'], label='Actual Demand', color='red', linestyle='--')
    plt.title("Inventory & Demand Flow (Single Episode)")
    plt.legend()
    plt.show()


def _plot_policy_diagnostic(fixed_avg, dyn_avg, config):
    """
    전체 에피소드의 평균 데이터를 바탕으로 정책 성과를 진단합니다.
    """
    # X축 (Day) 설정
    days = np.arange(len(fixed_avg['orders']))
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # (1) 가격 전략 비교 (평균 가격 추이)
    # dyn_avg['prices']는 (T, L) 형태의 행렬입니다.
    axes[0,0].plot(days, dyn_avg['prices'][:, 0], label="Dynamic: 1 Day Left (Avg)", color='red', alpha=0.8)
    axes[0,0].plot(days, dyn_avg['prices'][:, -1], label="Dynamic: Fresh (Avg)", color='green', alpha=0.8)
    axes[0,0].axhline(y=config.BASE_PRICE, color='black', linestyle='--', label="Fixed Price")
    axes[0,0].set_title("1. Avg Price Strategy (Expiring vs Fresh)")
    axes[0,0].set_ylabel("Price ($)")
    axes[0,0].legend()

    # (2) 주문량 비교 (평균 주문량 추이)
    axes[0,1].plot(days, fixed_avg['orders'], label="Fixed Order (Avg)", color='lightgrey', lw=2)
    axes[0,1].plot(days, dyn_avg['orders'], label="Dynamic Order (Avg)", color='orange', lw=2)
    axes[0,1].set_title("2. Avg Order Quantity Comparison")
    axes[0,1].set_ylabel("Quantity")
    axes[0,1].legend()

    # (3) 수익/비용 구조 분해 (전체 에피소드 평균 합계)
    labels = ['Revenue', 'Order Cost', 'Holding Cost', 'Waste Penalty']
    # 이미 main에서 N으로 나눈 평균값이므로 sum()으로 전체 기간 합계를 구합니다.
    f_vals = [np.sum(fixed_avg['revenue']), -np.sum(fixed_avg['order_costs']), 
              -np.sum(fixed_avg['holding_costs']), -np.sum(fixed_avg['waste_costs'])]
    d_vals = [np.sum(dyn_avg['revenue']), -np.sum(dyn_avg['order_costs']), 
              -np.sum(dyn_avg['holding_costs']), -np.sum(dyn_avg['waste_costs'])]
    
    x = np.arange(len(labels))
    axes[1,0].bar(x - 0.2, f_vals, 0.4, label='Fixed (Avg)', color='lightgrey')
    axes[1,0].bar(x + 0.2, d_vals, 0.4, label='Dynamic (Avg)', color='orange')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(labels)
    axes[1,0].set_title(f"3. Total Avg Profit/Cost Breakdown (N={config.N})")
    axes[1,0].axhline(0, color='black', linewidth=0.8)
    axes[1,0].legend()

    # (4) 일일 수익 흐름 (평균)
    axes[1,1].plot(days, fixed_avg['rewards'], label="Fixed Daily Profit (Avg)", color='lightgrey')
    axes[1,1].plot(days, dyn_avg['rewards'], label="Dynamic Daily Profit (Avg)", color='orange')
    axes[1,1].set_title("4. Avg Daily Profit Flow")
    axes[1,1].set_ylabel("Profit ($)")
    axes[1,1].axhline(0, color='red', linestyle=':', alpha=0.5)
    axes[1,1].legend()

    plt.tight_layout()
    plt.show()


def plot_policy_diagnostic(fixed_history, dynamic_history, config):
    """
    Fixed와 Dynamic의 행동 및 비용 구조를 직접 비교하는 대시보드
    """
    T = len(fixed_history['orders'])
    days = np.arange(T)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    ax_p = axes[0, 0]
    dyn_prices = np.array(dynamic_history['prices']) # (T, L)
    ax_p.plot(days, dyn_prices[:, 0], label="Dynamic: 1 Day Left (Expiring)", color='red', alpha=0.6)
    ax_p.plot(days, dyn_prices[:, -1], label="Dynamic: Fresh", color='darkgreen')
    ax_p.axhline(y=config.BASE_PRICE, color='black', linestyle='--', label="Fixed Price")
    ax_p.set_title("Price Strategy Comparison")
    ax_p.set_ylabel("Price ($)")
    ax_p.legend()

    ax_o = axes[0, 1]
    ax_o.plot(days, dynamic_history['orders'], label="Dynamic Order", color='blue', alpha=0.7)
    ax_o.plot(days, dynamic_history['demands'], label="Actual Demand", color='gray', linestyle=':', alpha=0.5)
    ax_o.set_title("Order Quantity vs Demand")
    ax_o.set_ylabel("Quantity")
    ax_o.legend()

    ax_c = axes[1, 0]
    labels = ['Revenue', 'Order Cost', 'Holding Cost', 'Waste Cost']
    
    def get_comp(hist):
        return [np.sum(hist['revenue']), -np.sum(hist['order_costs']), 
                -np.sum(hist['holding_costs']), -np.sum(hist['waste_costs'])]

    fixed_comp = get_comp(fixed_history)
    dyn_comp = get_comp(dynamic_history)
    
    x = np.arange(len(labels))
    width = 0.35
    ax_c.bar(x - width/2, fixed_comp, width, label='Fixed', color='skyblue')
    ax_c.bar(x + width/2, dyn_comp, width, label='Dynamic', color='orange')
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(labels)
    ax_c.set_title("Profit/Cost Breakdown (Total Sum)")
    ax_c.axhline(0, color='black', lw=1)
    ax_c.legend()

    ax_r = axes[1, 1]
    ax_r.plot(days, np.cumsum(fixed_history['revenue']) - np.cumsum(fixed_history['order_costs']), label="Fixed Profit", color='skyblue')
    ax_r.plot(days, np.cumsum(dynamic_history['revenue']) - np.cumsum(dynamic_history['order_costs']), label="Dynamic Profit", color='orange')
    ax_r.set_title("Cumulative Profit Growth")
    ax_r.set_ylabel("Net Profit ($)")
    ax_r.legend()

    plt.tight_layout()
    plt.show()