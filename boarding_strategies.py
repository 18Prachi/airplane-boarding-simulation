import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register

try:
    register(
        id="airplane-boarding-v0",
        entry_point="airplane_boarding:AirplaneEnv",  # ✅ Correct class name
    )
except Exception:
    pass



def make_env(rows=10, seats=5, render_mode=None):
    """Return a new airplane env instance."""
    return gym.make(
        "airplane-boarding-v0",
        num_of_rows=rows,
        seats_per_row=seats,
        render_mode=render_mode
    )


def random_strategy(env):
    """Pick random valid rows until boarding is done"""
    obs, _ = env.reset()
    total_reward, steps = 0, 0

    while True:
        masks = env.unwrapped.action_masks()
        valid_actions = [i for i, valid in enumerate(masks) if valid]
        if not valid_actions:
            break
        action = np.random.choice(valid_actions)

        obs, reward, terminated, _, _ = env.step(action)
        total_reward += reward
        steps += 1
        if terminated:
            break
    return steps, total_reward


def back_to_front(env):
    """Board passengers row by row, starting from the last row"""
    obs, _ = env.reset()
    total_reward, steps = 0, 0

    for row in reversed(range(env.unwrapped.num_of_rows)):
        while len(env.unwrapped.lobby.lobby_rows[row].passengers) > 0:
            obs, reward, terminated, _, _ = env.step(row)
            total_reward += reward
            steps += 1
            if terminated:
                return steps, total_reward
    return steps, total_reward


def front_to_back(env):
    """Board row by row, starting from the front"""
    obs, _ = env.reset()
    total_reward, steps = 0, 0

    for row in range(env.unwrapped.num_of_rows):
        while len(env.unwrapped.lobby.lobby_rows[row].passengers) > 0:
            obs, reward, terminated, _, _ = env.step(row)
            total_reward += reward
            steps += 1
            if terminated:
                return steps, total_reward
    return steps, total_reward


def wilma(env):
    """Window -> Middle -> Aisle (WilMA)"""
    obs, _ = env.reset()
    total_reward, steps = 0, 0

    # heuristic seat priority for a 5-seat row: [window left, window right, mid-left, mid-right, aisle]
    seats = env.unwrapped.seats_per_row
    # build seat priority dynamically: outermost seats first
    seat_order = []
    left = 0
    right = seats - 1
    while left <= right:
        seat_order.append(left)
        if right != left:
            seat_order.append(right)
        left += 1
        right -= 1

    passengers = []
    for row in env.unwrapped.lobby.lobby_rows:
        for p in row.passengers:
            passengers.append(p)

    # sort by seat priority then row (can tweak to back-to-front inside groups)
    passengers.sort(key=lambda p: (seat_order.index(p.seat_num % seats), p.row_num))

    for p in passengers:
        row = p.row_num
        obs, reward, terminated, _, _ = env.step(row)
        total_reward += reward
        steps += 1
        if terminated:
            break
    return steps, total_reward
