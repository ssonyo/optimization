"""
Visualization of Adaptive Market Planning results
-------------------------------------------------
Runs the AdaptiveMarketPlanning simulation for multiple theta_step values
and plots comparative results (reward evolution, learned order path, etc.)
"""

import numpy as np
import matplotlib.pyplot as plt
from AdaptiveMarketPlanningModel import AdaptiveMarketPlanningModel
from AdaptiveMarketPlanningPolicy import AdaptiveMarketPlanningPolicy


def run_experiment(cost, price, T, reward_type, trial_size, theta_step):
    """Run one full experiment and return results"""
    state_names = ["order_quantity", "counter"]
    init_state = {"order_quantity": 0, "counter": 0}
    decision_names = ["step_size"]

    # Model + Policy
    M = AdaptiveMarketPlanningModel(state_names, decision_names, init_state, T, reward_type, price, cost)
    P = AdaptiveMarketPlanningPolicy(M, theta_step)

    rewards = []
    learning_paths = []

    for ite in range(trial_size):
        reward, learning_list = P.run_policy()
        M.learning_list = []  # reset learning log
        rewards.append(reward)
        learning_paths.append(learning_list)

    rewards = np.array(rewards)
    n = np.arange(1, trial_size + 1)

    # cumulative average
    cum_avg = rewards.cumsum() / n

    # 보상 유형에 따른 평균 계산 방식
    if reward_type == "Cumulative":
        rewards = rewards / T
        cum_avg = cum_avg / T

    return rewards, cum_avg, learning_paths


def plot_rewards(results_dict, T, reward_type):
    """Plot cumulative average reward and reward per iteration for all theta_steps"""
    plt.figure(figsize=(12, 5))

    # ---- Cum avg reward ----
    plt.subplot(1, 2, 1)
    for theta_step, res in results_dict.items():
        _, cum_avg, _ = res
        plt.plot(cum_avg, label=f"θ_step={theta_step}")
    plt.title("Cumulative average reward")
    plt.xlabel("Iteration")
    plt.ylabel("USD")
    plt.legend()

    # ---- Per iteration reward ----
    plt.subplot(1, 2, 2)
    for theta_step, res in results_dict.items():
        rewards, _, _ = res
        plt.plot(rewards, label=f"θ_step={theta_step}")
    plt.title("Reward per iteration")
    plt.xlabel("Iteration")
    plt.ylabel("USD")
    plt.legend()

    plt.suptitle(f"Reward type: {reward_type}, T={T}")
    plt.tight_layout()
    plt.show()


def plot_learning_paths(results_dict, cost, price, T, reward_type):
    """Plot one sample learning path (order quantity) for each theta_step"""
    optimal_q = -np.log(cost / price) * 100
    time = np.arange(T)

    plt.figure(figsize=(8, 6))
    for theta_step, res in results_dict.items():
        _, _, learning_paths = res
        sample_idx = np.random.randint(len(learning_paths))
        q_path = learning_paths[sample_idx]
        plt.plot(time, q_path, label=f"θ_step={theta_step}")

    plt.axhline(optimal_q, color="black", linestyle="--", label="Analytical optimum")
    plt.title(f"Learned order quantity trajectories ({reward_type} reward)")
    plt.xlabel("Time (months)")
    plt.ylabel("Order quantity")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # ---- 실험 파라미터 ----
    cost = 20
    trial_size = 100
    price = 26
    theta_steps = [2, 5, 10, 20, 50]
    T = 24
    reward_type = "Terminal"

    if reward_type not in ["Cumulative", "Terminal"]:
        print("Invalid input. Defaulting to 'Cumulative'.")
        reward_type = "Cumulative"

    # ---- Run experiments ----
    results = {}
    for theta_step in theta_steps:
        print(f"\nRunning experiment for θ_step = {theta_step} ({reward_type} reward)")
        results[theta_step] = run_experiment(cost, price, T, reward_type, trial_size, theta_step)

    # ---- Visualization ----
    plot_rewards(results, T, reward_type)
    plot_learning_paths(results, cost, price, T, reward_type)

    # ---- Summary ----
    print("\n=== Average rewards summary ===")
    for theta_step, res in results.items():
        rewards, cum_avg, _ = res
        print(f"θ_step={theta_step:>4} | Avg reward per iteration: {cum_avg[-1]:.4f}")


if __name__ == "__main__":
    main()
