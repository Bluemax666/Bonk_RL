# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 03:46:52 2022

@author: maxim
"""
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver import ActionChains
import pyautogui
from utils.mouseFunctions import setMousePos, leftClick
from utils.grabScreen import grabScreen
from collections import deque
from sb3_contrib import TQC
from gym import spaces
import numpy as np
import pygame
import time
import cv2

class Env():
    def __init__(self, p1_name="player1", p2_name="player2", update_opponent_every=200):
        
        self.p1_name = p1_name
        self.p2_name = p2_name
        self.p1_color = "blue"
        self.p2_color = "red"
        
        self.host = BonkPlayer(self.p1_name, self.p1_color)
        self.host.createGame()
        time.sleep(2)
        self.opponent = BonkPlayer(self.p2_name, self.p2_color)
        self.opponent.joinGame()
        self.opponent.switchWindow()
        self.host.setMap()
        self.host.startGame()
        
        self.win_reward = 50
        self.FPS_limit = 15
        self.last_time = time.time()
        self.episode_duration = 60 
        
        self.update_opponent_every = update_opponent_every
        self.opponent_start_episode = 10
        
        self.nb_actions = 3
        self.len_observation = 16
        
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.len_observation,), dtype=np.float32)
        print("Observation space:", self.observation_space)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.nb_actions,), dtype=np.float32)
        print("Action space:", self.action_space)
        self.reward_range = 1
        print("reward range:", self.reward_range)
        self.metadata = ""
        
        self.model = None
        self.opponent_model = None
        self.episode_count = 1
        self.n_cp_opponent = 0
        
    def setModel(self, model):
        self.model = model
        self.model.save(f"cp/opponent_{self.n_cp_opponent}")
        self.opponent_model = TQC.load(f"cp/opponent_{self.n_cp_opponent}")
        
    def reset(self):
        self.episode_count += 1
        
        if self.episode_count % self.update_opponent_every == 0:
            self.n_cp_opponent += 1
            self.model.save(f"cp/opponent_{self.n_cp_opponent}")
            self.opponent_model = TQC.load(f"cp/opponent_{self.n_cp_opponent}")
        
        self.restartGame()
        
        self.host.resetPosInfo()
        new_state_p1, new_state_p2 = self.host.getObservation()
        self.last_time = time.time()
        self.ep_start_time = time.time()
        
        self.host.applyControls([0,0,0,0,0])
        self.opponent.applyControls([0,0,0,0,0])
        
        self.state_p2 = new_state_p2
        self.lt = time.time()
        
        return new_state_p1
        
        
    def reset2(self):
        self.episode_count += 1
        
        self.restartGame()
        
        self.host.resetPosInfo()
        new_state_p1, new_state_p2 = self.host.getObservation()
        self.last_time = time.time()
        self.ep_start_time = time.time()
        
        self.host.applyControls([0,0,0,0,0])
        self.opponent.applyControls([0,0,0,0,0])
        
        self.state_p2 = new_state_p2
        self.lt = time.time()
        
        return new_state_p1, new_state_p2
    
    def restartGame(self):
        self.host.exitGame()
        leftClick(681, 726)
        time.sleep(0.5)
        self.host.switchWindow()
        time.sleep(0.1)

        self.opponent.scrollUp(5)
        
        time.sleep(0.1)
        leftClick(473, 564)
        time.sleep(0.1)
        leftClick(473, 564)
        time.sleep(0.1)
        
        self.host.switchWindow()
        time.sleep(0.25)
        
        self.host.scrollUp(5)
        
        leftClick(473, 564)
        time.sleep(0.1)
        
        
    def step(self, action_p1):
        
        if self.episode_count >= self.opponent_start_episode:
            action_p2, _states = self.opponent_model.predict(self.state_p2, deterministic=True)
        else:
            action_p2 = [0,0,0]
        
        controls_p1 = self.getControls(action_p1)
        self.host.applyControls(controls_p1)
            
        controls_p2 = self.getControls(action_p2)
        self.opponent.applyControls(controls_p2)
        
        self.limitFPS()
        
        self.host.detectBalls()
        new_state_p1, new_state_p2 = self.host.getObservation()
        self.state_p2 = new_state_p2
        
        (speed_x_p1, speed_y_p1), (speed_x_p2, speed_y_p2) = self.host.getSpeed()
        alive_p1, alive_p2 = self.host.player1.alive, self.host.player2.alive
        
        reward_p1 = speed_x_p1/5000 + speed_x_p2/5000
        reward_p2 = speed_x_p1/5000 + speed_x_p2/5000
        
        done = False
        if alive_p1 ^ alive_p2:
            done = True
        
        if done:
            if alive_p1:
                reward_p1 += self.win_reward
                reward_p2 -= self.win_reward/3
            
            elif alive_p2:
                reward_p2 += self.win_reward
                reward_p1 -= self.win_reward/3
                
        if time.time() - self.ep_start_time > self.episode_duration:
            done = True
            
        #self.showFPS()
        
        return new_state_p1, reward_p1, done, {}
        
        
    def step2(self, action_p1, action_p2):
        
        if action_p1 is not None: 
            controls_p1 = self.getControls(action_p1)
            self.host.applyControls(controls_p1)
            
        if action_p2 is not None:
            controls_p2 = self.getControls(action_p2)
            self.opponent.applyControls(controls_p2)
        
        self.limitFPS()
        
        self.host.detectBalls()
        new_state_p1, new_state_p2 = self.host.getObservation()
        self.state_p2 = new_state_p2
        
        (speed_x_p1, speed_y_p1), (speed_x_p2, speed_y_p2) = self.host.getSpeed()
        alive_p1, alive_p2 = self.host.player1.alive, self.host.player2.alive
        
        reward_p1 = speed_x_p1/5000 + speed_x_p2/5000
        reward_p2 = speed_x_p1/5000 + speed_x_p2/5000
        
        done = False
        if alive_p1 ^ alive_p2:
            done = True
        
        if done:
            if alive_p1:
                reward_p1 += self.win_reward
                reward_p2 -= self.win_reward/3
            
            elif alive_p2:
                reward_p2 += self.win_reward
                reward_p1 -= self.win_reward/3
                
        if time.time() - self.ep_start_time > self.episode_duration:
            done = True
            
        #self.showFPS()
        
        return (new_state_p1, new_state_p2), (reward_p1, reward_p2), done, {}

    def showFPS(self):
        nt = time.time()
        print("{} fps".format(round(1/(nt-self.lt),2)))
        self.lt = nt
        
    def limitFPS(self):
        if self.FPS_limit != -1:
            mil_delay = int(((1/self.FPS_limit) - (time.time()-self.last_time))*1000)
            if mil_delay > 0 and mil_delay < 1000/self.FPS_limit:
                pygame.time.delay(mil_delay)
                
            self.last_time = time.time()
            
    
    def getControls(self, action):
        thresh = 0.5
        """action : [down-up], [left-right], [notheavy - heavy] """
        """controls : [up, down, left, right, heavy]"""
        controls = [0, 0, 0, 0, 0]
        if action[0] > thresh:
            controls[0] = 1
        elif action[0] < -thresh:
            controls[1] = 1
            
        if action[1] > thresh:
            controls[3] = 1
        elif action[1] < -thresh:
            controls[2] = 1
            
        if action[2] > thresh:
            controls[4] = 1
        
        return controls
    
    def action_space_sample(self):
        return np.random.rand(3)*2 - 1
    
    def render(self):
        pass
        
     
class BallInfo():
    def __init__(self, avg_len):
        self.avg_len = avg_len
        
        self.position_list = deque(maxlen=self.avg_len)
        self.velocity_list = deque(maxlen=self.avg_len)
        self.acceleration_list = deque(maxlen=self.avg_len)
        
        self.alive = False
        self.pos = (-1,-1)
        self.velocity = (0,0)
        self.acceleration = (0,0)
        
    def updatePos(self, pos, frame_rate):
        self.position_list.append(pos)
        if len(self.position_list) == self.avg_len:
            avg_velocity = (np.array(self.position_list[-1]) - np.array(self.position_list[-self.avg_len])) / (self.avg_len-1) * frame_rate
            self.velocity_list.append(tuple(avg_velocity))
        
        if len(self.velocity_list) == self.avg_len:
            avg_accel = (np.array(self.velocity_list[-1]) - np.array(self.velocity_list[-self.avg_len])) / (self.avg_len-1) * frame_rate
            self.acceleration_list.append(tuple(avg_accel))
            
            self.pos = pos
            self.velocity = self.velocity_list[-1]
            self.acceleration = self.acceleration_list[-1]
            
    def resetPos(self, init_pos):
        self.position_list = deque(maxlen=self.avg_len)
        self.velocity_list = deque(maxlen=self.avg_len)
        self.acceleration_list = deque(maxlen=self.avg_len)
        
        self.alive = False
        self.pos = init_pos
        self.velocity = (0,0)
        self.acceleration = (0,0)
            

class BonkPlayer():
    def __init__(self, name, color, launch_game=True):
        
        self.name = name
        self.color = color
        if launch_game:
            self.chrome_options = Options()
            self.chrome_options.add_argument("--window-size=1920x1080")
            self.chrome_options.add_argument("--headless")
            self.driver = webdriver.Chrome('chromedriver')
            self.driver.get('https://bonk.io')
        
        self.avg_len = 5
        self.player1 = BallInfo(self.avg_len)
        self.player2 = BallInfo(self.avg_len)
        
        self.capture_coords = [200,380,750,760]
        self.w, self.h = 750 - 200, 760 - 380
        self.cx, self.cy = 273, 184
        
        self.frame_rate = 30
        self.last_time = time.time()
        
        self.controls = [Keys.UP, Keys.DOWN, Keys.LEFT, Keys.RIGHT, Keys.SHIFT]
        self.last_controls = [0, 0, 0, 0, 0]
        
        time.sleep(2)
        
        if launch_game:
            leftClick(719, 799)
            time.sleep(1)
            leftClick(347, 632)
            time.sleep(1)
            
            leftClick(470, 539)
            time.sleep(0.5)
            if self.color == "blue":
                leftClick(205, 510)
            
            elif self.color == "red":
                leftClick(363, 589)
            
            leftClick(470, 665)
            pyautogui.keyDown('ctrl')
            pyautogui.keyDown('a')
            pyautogui.keyUp('ctrl')
            pyautogui.keyUp('a')
            pyautogui.press('backspace')
            pyautogui.typewrite(self.name, interval=0.05)
            
            leftClick(475, 721)
            time.sleep(0.5)
            
            leftClick(873, 23)
            setMousePos(1448, 302)
            time.sleep(0.5)
            leftClick(1446, 303)
            time.sleep(0.5)
            leftClick(1111, 795)
            time.sleep(0.5)
            leftClick(1089, 815)
            time.sleep(0.5)
            leftClick(1021, 906)
            time.sleep(0.5)
            leftClick(1848, 16)
            time.sleep(1)
            
        
    def getPos(self, frame, color):
           
        HSV_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2HSV)
        mask_range = np.array([5,5,5])
        mask_lower = np.clip(color - mask_range, 0, 255)
        mask_upper =  np.clip(color + mask_range, 0, 255)
        ball_mask = cv2.inRange(HSV_frame, np.array(mask_lower, np.uint8) , np.array(mask_upper, np.uint8))
        
        M = cv2.moments(ball_mask)
        is_ball, ball_x, ball_y = False, -1, -1
        if M["m00"] != 0:
            is_ball = True
            ball_x = round(M["m10"] / M["m00"], 2)
            ball_y = round(M["m01"] / M["m00"], 2)
            
        return is_ball, (ball_x, ball_y)
    
    def resetPosInfo(self):
        p1_ball_pos, p2_ball_pos = self.detectBalls()
        self.player1.resetPos(p1_ball_pos)
        self.player2.resetPos(p2_ball_pos)
    
    def detectBalls(self):
        frame = grabScreen(self.capture_coords)
        blue = np.array([93, 255, 212])
        red = np.array([7,239,191])
        
        p1_is_ball, p1_ball_pos = self.getPos(frame, blue)
        p2_is_ball, p2_ball_pos = self.getPos(frame, red)
        
        if p1_is_ball:
            self.player1.alive = True
            self.player1.updatePos(p1_ball_pos, self.frame_rate)    
        else:
            self.player1.alive = False
            
        if p2_is_ball:
            self.player2.alive = True
            self.player2.updatePos(p2_ball_pos, self.frame_rate)
        else:
            self.player2.alive = False
            
        self.updateFramerate()
        
        return (p1_ball_pos, p2_ball_pos)
        
    def getPlayerObs(self, player, opponent, mirror=False):
        speed_div = 300
        accel_div = 400
        obs = []
        obs.append((player.pos[0]-self.cx) / (self.w/2))
        obs.append((player.pos[1]-self.cy) / (self.h/2))
        obs.append(player.velocity[0]/speed_div)
        obs.append(player.velocity[1]/speed_div)
        obs.append(player.acceleration[0]/accel_div)
        obs.append(player.acceleration[1]/accel_div)
        
        obs.append((opponent.pos[0]-self.cx) / (self.w/2))
        obs.append((opponent.pos[1]-self.cy) / (self.h/2))
        obs.append(opponent.velocity[0]/speed_div)
        obs.append(opponent.velocity[1]/speed_div)
        obs.append(opponent.acceleration[0]/accel_div)
        obs.append(opponent.acceleration[1]/accel_div)
        
        obs.append((player.pos[0]-opponent.pos[0]) / (self.w))
        obs.append((player.pos[1]-opponent.pos[1]) / (self.h))
        obs.append((player.velocity[0] - opponent.velocity[0])/speed_div)
        obs.append((player.velocity[1] - opponent.velocity[1])/speed_div)
        
        numpy_obs = np.array(obs, dtype=np.float32)
        numpy_obs = np.clip(numpy_obs, -2 , 2)
        
        if mirror:
            numpy_obs *= -1
        
        return numpy_obs 
        
    
    def getObservation(self):
        player1_obs = self.getPlayerObs(self.player1, self.player2)
        player2_obs = self.getPlayerObs(self.player2, self.player1)
        
        return player1_obs, player2_obs
    
    def getSpeed(self):
        return ((abs(self.player1.velocity[0]), abs(self.player1.velocity[1])),
                (abs(self.player2.velocity[0]), abs(self.player2.velocity[1])))
        
    def updateFramerate(self):
        new_time = time.time()
        
        current_FPS = max(1, 1/(new_time-self.last_time))
        
        self.frame_rate = 0.8 * self.frame_rate + 0.2 * current_FPS
        self.last_time = new_time    
        
    def applyControls(self, controls):
        for i in range(len(controls)):
            if self.last_controls[i] == 0 and controls[i] > 0:
                self.pressControlIndex(i)
                
            elif self.last_controls[i] > 0 and controls[i] == 0:
                self.releaseControlIndex(i)
        
        self.last_controls = controls
        
    def pressControlIndex(self, index):
        ActionChains(self.driver)\
             .key_down(self.controls[index])\
             .perform()
             
    
    def releaseControlIndex(self, index):
        ActionChains(self.driver)\
             .key_up(self.controls[index])\
             .perform()    
        
    def pressUp(self):
        ActionChains(self.driver)\
            .key_down(Keys.UP)\
            .perform()
            
    def releaseUp(self):
        ActionChains(self.driver)\
            .key_up(Keys.UP)\
            .perform()
            
    def scrollUp(self, n=5):
        leftClick(444, 980)
        for i in range(n):
            self.applyControls([0,0,0,0,0])
            time.sleep(0.1)
            self.applyControls([1,0,0,0,0])
            
        time.sleep(0.1)
        self.applyControls([0,0,0,0,0])
    
    def pressDown(self):
        ActionChains(self.driver)\
            .key_down(Keys.DOWN)\
            .perform()
            
    def pressLeft(self):
        ActionChains(self.driver)\
            .key_down(Keys.LEFT)\
            .perform()
            
    def pressRight(self):
        ActionChains(self.driver)\
            .key_down(Keys.RIGHT)\
            .perform()
            
    def pressHeavy(self):
        ActionChains(self.driver)\
            .key_down(Keys.SHIFT)\
            .perform()
    
    def scrollUP(self):
        ActionChains(self.driver)\
        .scroll_by_amount(0, -100)\
        .perform()
        
    def createGame(self):
        leftClick(467, 540)
        time.sleep(0.5)
        leftClick(251, 700)
        time.sleep(0.5)
        leftClick(468, 527)
        pyautogui.typewrite('OUIOUIBAGUETTE', interval=0.05)
        # leftClick(490, 583)
        # pyautogui.keyDown('return')
        # pyautogui.typewrite('3', interval=0.05)
        leftClick(538, 720)
        time.sleep(1)
        
        
    def setMap(self):
        leftClick(590, 683)
        time.sleep(0.5)
        leftClick(306, 487)
        time.sleep(1)
        leftClick(306, 634)
        time.sleep(1)
        leftClick(567, 484)
        time.sleep(0.2)
        pyautogui.typewrite('1v1 1v1 1v1', interval=0.05)
        time.sleep(0.5)
        pyautogui.press('enter')
        time.sleep(2)
        leftClick(490, 589)
        time.sleep(1.5)
        
    def startGame(self):
        leftClick(681, 726)
        time.sleep(5.5)
        
    def exitGame(self):
        setMousePos(715, 405)
        time.sleep(0.2)
        leftClick(720, 400)
        time.sleep(0.4)
        
    def joinGame(self):
        leftClick(467, 540)
        time.sleep(0.5)
        leftClick(485, 491)
        time.sleep(0.5)
        leftClick(695, 701)
        time.sleep(0.5)
        pyautogui.typewrite('OUIOUIBAGUETTE', interval=0.05)
        leftClick(543, 626)
        time.sleep(1)
        
    def switchWindow(self):
        pyautogui.keyDown('alt')
        pyautogui.keyDown('tab')
        pyautogui.keyUp('alt')
        pyautogui.keyUp('tab')
        time.sleep(0.5)