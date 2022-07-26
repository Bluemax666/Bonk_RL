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

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        
    def _on_step(self) -> bool:
        if keyboard.is_pressed('p'):
            return False
        else:
            return True
        
custom_callback = CustomCallback()

time.sleep(3)
env = Env(update_opponent_every=400)


# policy_kwargs = dict(n_critics=2, n_quantiles=25, net_arch=dict(pi=[256, 256], qf=[256, 256, 256]))
# model = TQC("MlpPolicy", env, gamma=0.9925, top_quantiles_to_drop_per_net=2, batch_size=192,
#             use_sde=True, policy_kwargs=policy_kwargs, verbose=1, device='cpu')

model = TQC.load("runs/bonk02", env=env)
env.setModel(model)

model.learn(total_timesteps=600_000, log_interval=25, callback=custom_callback)
model.save("runs/bonk03")