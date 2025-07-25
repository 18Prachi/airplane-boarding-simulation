import gymnasium as gym
from airplane_boarding import AirplaneEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib.common.maskable.callbacks import  MaskableEvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold

import os

model_dir = "models"
log_dir = "logs"

def train():
    try:
        env = make_vec_env(AirplaneEnv, n_envs=12, env_kwargs={"num_of_rows":10, "seats_per_row":5}, vec_env_cls=SubprocVecEnv)

        # Increase ent_coef to encourage exploration, this resulted in a better solution.
        model = MaskablePPO('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir, ent_coef=0.1)  # device  = 'cuda' if NVIDIA GPU else 'cpu'

        eval_callback = MaskableEvalCallback(
            env,
            eval_freq=10_000,
            # callback_on_new_best = StopTrainingOnRewardThreshold(reward_threshold=???, verbose=1)
            # callback_after_eval  = StopTrainingOnNoModelImprovement(max_no_improvement_evals=???, min_evals=???, verbose=1)
            verbose=1,
            best_model_save_path=os.path.join(model_dir, 'MaskablePPO'),
        )

        """
        total_timesteps: pass in a very large number to train (almost) indefinitely.
        callback: pass in reference to a callback fuction above
        """
        model.learn(total_timesteps=int(1e10), callback=eval_callback)
    except Exception as e:
        print(f"Error during training: {e}")

def test(model_name, render=True):
    try:
        env = gym.make('airplane-boarding-v0', num_of_rows=10, seats_per_row=5, render_mode='human' if render else None)

        # Load model
        model = MaskablePPO.load(f'models/MaskablePPO/{model_name}', env=env)

        rewards = 0
        # Run a test
        obs, _ = env.reset()
        terminated = False

        while True:
            try:
                action_masks = get_action_masks(env)
                action, _ = model.predict(observation=obs, deterministic=True, action_masks=action_masks) # Turn on deterministic, so predict always returns the same behavior
                obs, reward, terminated, _, _ = env.step(action)
                rewards += reward

                if terminated:
                    break
            except Exception as e:
                print(f"Error during test step: {e}")
                break

        print(f"Total rewards: {rewards}")
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == '__main__':
    train()
    # test("best_model")


