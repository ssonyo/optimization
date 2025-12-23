"""
Visualization for Adaptive Market Planning (case e)
---------------------------------------------------
Uses ParametricModel where Ft = p_{t+1} * min(xt, Wt+1) - c xt
Plots comparative rewards and learning paths across multiple θ_step values
"""

import numpy as np
import matplotlib.pyplot as plt
from ParametricModel import ParametricModel


def run_experiment(T, reward_type, cost, price_low, price_high, price_process, theta_step, trial_size):
    """Run multiple trials and collect rewards + learning trajectories"""
    state_names = ["counter", "price", "theta"]
    init_state = {"counter": 0, "price": 5.0, "theta": np.array([20.0, -2.0, 5.0], dtype=float)}
    x_names = ["step_size"]

    rewards = []
    learning_paths = []

    for _ in range(trial_size):
        model = ParametricModel(
            state_names=state_names,
            x_names=x_names,
            s_0=init_state,
            T=T,
            reward_type=reward_type,
            cost=cost,
            price_low=price_low,
            price_high=price_high,
            price_process=price_process,
            seed=np.random.randint(1e6)
        )

        for _ in range(T):
            step_size = float(theta_step) / (1.0 + float(model.state.counter))
            decision = model.build_decision({"step_size": step_size})
            model.step(decision)

        rewards.append(model.obj)
        learning_paths.append(np.array([x[0] + x[1] + x[2] for x in model.learning_list]) if model.learning_list else np.zeros(T))

    rewards = np.array(rewards)
    n = np.arange(1, trial_size + 1)
    cum_avg = rewards.cumsum() / n

    if reward_type == "Cumulative":
        rewards = rewards / T
        cum_avg = cum_avg / T

    return rewards, cum_avg, learning_paths


def plot_rewards(results_dict, T, reward_type):
    """Plot cumulative average reward and reward per iteration for all θ_steps"""
    plt.figure(figsize=(12, 5))

    # ---- Cumulative average reward ----
    plt.subplot(1, 2, 1)
    for theta_step, res in results_dict.items():
        _, cum_avg, _ = res
        plt.plot(cum_avg, label=f"θ_step={theta_step}")
    plt.title("Cumulative average reward")
    plt.xlabel("Iteration")
    plt.ylabel("USD")
    plt.legend()

    # ---- Reward per iteration ----
    plt.subplot(1, 2, 2)
    for theta_step, res in results_dict.items():
        rewards, _, _ = res
        plt.plot(rewards, label=f"θ_step={theta_step}")
    plt.title("Reward per iteration")
    plt.xlabel("Iteration")
    plt.ylabel("USD")
    plt.legend()

    plt.suptitle(f"Reward type: {reward_type}, T={T} (price revealed next period)")
    plt.tight_layout()
    plt.show()


def plot_learning_paths(results_dict, T, reward_type):
    """Plot one sample θ trajectory (proxy for learned order) per θ_step"""
    time = np.arange(T)
    plt.figure(figsize=(8, 6))
    for theta_step, res in results_dict.items():
        _, _, learning_paths = res
        if not learning_paths:
            continue
        q_path = learning_paths[np.random.randint(len(learning_paths))]
        plt.plot(time, q_path, label=f"θ_step={theta_step}")

    plt.title(f"Learning trajectories (reward={reward_type}, price=p_(t+1))")
    plt.xlabel("Time step")
    plt.ylabel("θ sum (proxy for order level)")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # ---- 실험 파라미터 ----
    cost = 4.0
    price_low = 1.0
    price_high = 10.0
    price_process = "RW"
    T = 24
    reward_type = "Terminal"     # (e)
    trial_size = 100
    theta_steps = [2, 5, 10, 20, 50]
    theta_steps = [0.2, 0.5, 1, 2, 5]

    # ---- Run experiments ----
    results = {}
    for theta_step in theta_steps:
        print(f"\nRunning (e) case: θ_step={theta_step} ({reward_type})")
        results[theta_step] = run_experiment(
            T, reward_type, cost, price_low, price_high, price_process, theta_step, trial_size
        )

    # ---- Visualization ----
    plot_rewards(results, T, reward_type)
    plot_learning_paths(results, T, reward_type)

    # ---- Summary ----
    print("\n=== (e) Average rewards summary ===")
    for theta_step, res in results.items():
        rewards, cum_avg, _ = res
        print(f"θ_step={theta_step:>7} | Final avg reward: {cum_avg[-1]:.4f}")


if __name__ == "__main__":
    main()
