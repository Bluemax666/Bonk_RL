# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 02:49:45 2022

@author: Maxime
"""
from bonk_env import Env
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import BaseCallback
import keyboard
import time

time.sleep(3)
env = Env(save_opponent_every=1000, n_start_opponent=39)

policy_kwargs = dict(n_critics=3, n_quantiles=25, net_arch=dict(pi=[256, 256, 256], qf=[512, 512, 512]))
model = TQC("MlpPolicy", env, gamma=0.995, top_quantiles_to_drop_per_net=2, batch_size=256,
            use_sde=True, policy_kwargs=policy_kwargs, verbose=1, device='cuda')

env.setModel(model)

model.learn(total_timesteps=2_000_000, log_interval=25)
model.save("runs/bonk01")
