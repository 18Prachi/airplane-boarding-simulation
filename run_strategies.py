import numpy as np
import matplotlib.pyplot as plt
from boarding_strategies import make_env, random_strategy, back_to_front, front_to_back, wilma

# Strategies to compare
strategies = {
    "Random": random_strategy,
    "Back-to-Front": back_to_front,
    "Front-to-Back": front_to_back,
    "WilMA": wilma,
}

def evaluate_strategy(strategy_func, runs=10, rows=10, seats=5):
    """Run one strategy multiple times and return avg steps + reward"""
    steps_list, reward_list = [], []
    for _ in range(runs):
        env = make_env(rows=rows, seats=seats)
        steps, reward = strategy_func(env)
        env.close()
        steps_list.append(steps)
        reward_list.append(reward)
    return np.mean(steps_list), np.mean(reward_list)

def main():
    results = {}

    for name, strategy in strategies.items():
        avg_steps, avg_reward = evaluate_strategy(strategy, runs=20, rows=10, seats=5)
        results[name] = (avg_steps, avg_reward)
        print(f"{name:15s} -> Steps: {avg_steps:.2f}, Reward: {avg_reward:.2f}")

    # Extract data for plotting
    names = list(results.keys())
    steps = [results[name][0] for name in names]
    rewards = [results[name][1] for name in names]

    # Plot side-by-side charts
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Average Steps
    axs[0].bar(names, steps, color="skyblue")
    axs[0].set_ylabel("Average Steps (lower is better)")
    axs[0].set_title("Boarding Strategy - Avg Steps")

    # Average Rewards
    axs[1].bar(names, rewards, color="lightgreen")
    axs[1].set_ylabel("Average Reward (higher is better)")
    axs[1].set_title("Boarding Strategy - Avg Reward")

    plt.suptitle("Boarding Strategy Comparison")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
