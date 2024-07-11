import gym
import pandas as pd
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from torch.utils.tensorboard import SummaryWriter


class TensorboardCallback(BaseCallback):
    def __init__(self, writer, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.writer = writer

    def _on_step(self) -> bool:
        self.writer.add_scalar("reward", self.locals["rewards"][0], self.num_timesteps)
        return True


hyperparams = [
    {'learning_rate': 0.01, 'batch_size': 128, 'gamma': 0.85},
    {'learning_rate': 0.001, 'batch_size': 128, 'gamma': 0.9},
    {'learning_rate': 0.01, 'batch_size': 64, 'gamma': 0.99},
]

writer = SummaryWriter()

for params in hyperparams:

    for i in range(10):
        vec_env = make_vec_env('Pendulum-v1', n_envs=1)
        model = PPO('MlpPolicy', vec_env, **params, verbose=1, tensorboard_log="ppo/")

        logger = configure("results/", ["csv"])
        model.set_logger(logger)
        callback = TensorboardCallback(writer)

        model.learn(total_timesteps=50000, log_interval=1, callback=callback)

        output = pd.read_csv("results/progress.csv", sep=',')
        ep_rew = output['rollout/ep_rew_mean'].to_numpy()
        ep_time = output['time/time_elapsed'].to_numpy()

        print(f'Time for {i + 1} experiment: {ep_time.max()}')

writer.close()
