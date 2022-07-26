# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 02:49:45 2022

@author: Maxime
"""
from bonk_env import Env
from sb3_contrib import TQC
import time

time.sleep(3)
env = Env()

policy_kwargs = dict(n_critics=2, n_quantiles=25, net_arch=dict(pi=[256, 256], qf=[256, 256, 256]))
model = TQC("MlpPolicy", env, gamma=0.9925, top_quantiles_to_drop_per_net=2, batch_size=192,
            use_sde=True, policy_kwargs=policy_kwargs, verbose=1)

env.setModel(model)

model.learn(total_timesteps=600_000, log_interval=25)
model.save("runs/bonk01")
