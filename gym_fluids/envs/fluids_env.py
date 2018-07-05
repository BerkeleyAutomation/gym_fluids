import fluids
import pygame
import numpy as np
import gym
from gym import spaces


OBS_W = 400
TIME_LIMIT = 600

FLUIDS_ARGS = {"visualization_level"      :1,
               "render_on"                :False,
               "background_cars"          :10,
               "controlled_cars"          :1,
               "background_peds"          :5,
               "fps"                      :30,
               "obs_args"                 :{"obs_dim":OBS_W},
               "obs_space"                :fluids.OBS_BIRDSEYE,
               "background_control"       :fluids.BACKGROUND_CSP}

class FluidsEnv(gym.Env):
    def __init__(self):
        self.fluidsim = fluids.FluidSim(**FLUIDS_ARGS)
        self.action_space = spaces.Box(np.array([-1, -1]), np.array([1, 1]))
        self.observation_space = spaces.Box(low=0, high=255, shape=(OBS_W, OBS_W, 3), dtype=np.uint8)

    def reset(self):
        del(self.fluidsim)
        self.fluidsim = fluids.FluidSim(**FLUIDS_ARGS)
        obs, rew = self.fluidsim.step()
        car_keys = list(self.fluidsim.get_control_keys())
        return obs[car_keys[0]].get_array()


    def step(self, action):
        car_keys = list(self.fluidsim.get_control_keys())
        actions = {car_keys[0]: fluids.SteeringAction(action[0], action[1])}
        obs, reward_step = self.fluidsim.step(actions)

        done = self.fluidsim.run_time() > TIME_LIMIT
        obs = obs[car_keys[0]].get_array()
        return obs, reward_step, done, {}


    def render(self, mode='human'):
        self.fluidsim.render()
