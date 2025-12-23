import numpy as np

class StatsCollector:
    def __init__(self):
        self.results = {"Fixed": [], "Dynamic": []}

    def add_episode_result(self, policy_name, total_reward, total_waste, total_sales):
        waste_rate = total_waste / total_sales if total_sales > 0 else 0
        self.results[policy_name].append({
            "reward": total_reward,
            "waste_rate": waste_rate
        })

    def print_summary(self):
        print("\n" + "="*45)
        print(f"{'Policy':<12} | {'Avg Profit':<12} | {'Waste Rate':<10}")
        print("-" * 45)
        for name, data in self.results.items():
            if not data: continue
            avg_profit = np.mean([d['reward'] for d in data])
            avg_waste = np.mean([d['waste_rate'] for d in data])
            print(f"{name:<12} | {avg_profit:12.2f} | {avg_waste:10.2%}")
        print("="*45 + "\n")