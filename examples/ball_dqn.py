import gym
import gym_ballenv
import numpy as np 

import matplotlib
import matplotlib.pyplot as plt 

from collections import namedtuple

from itertools import count

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 


class DQN(nn.Module):

	def __init__(self):
		super(DQN ,self).__init__()
		self.fc1 = nn.f

def assert_arguments(args):

    assert len(args.obstacle_speed)== args.dynamic_obstacles, "The length of the list of obstacle_speed does not match the no. of dynamic obstacles"
    assert len(args.obs_goal_position)==args.dynamic_obstacles , "The length of the list of obstacle_goal_position does not match the no. of dynamic obstacles"
    assert len(args.static_thresholds)==2 , "The length of the list of static_thresholds is not equal to 2"
    assert len(args.dynamic_thresholds)==2 , "The length of the list of dynamic_thresholds is not equal to 2"     #except Exception as error:
    assert len(args.static_penalty)==2 , "The length of the list of static_penalty is not equal to 2"    
    assert len(args.dynamic_penalty)==2 , "The length of the list of dynamic_penalty is not equal to 2"
    #   print error
    #    exit()

args = read_arguments()

assert_arguments(args)
#print('sssss',type(args.obs_goal_position[1]))

#print(args)
env = gym.make('gymball-v0') # create the environment
#print(type(env.unwrapped))
env.unwrapped.customize_environment(args)


