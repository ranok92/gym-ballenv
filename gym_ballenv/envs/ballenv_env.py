import gym
import math
import numpy as np
from gym import error ,spaces, utils
from gym.utils import seeding

class BallEnv(gym.Env):

	metadata = {'render.modes': ['human' , 'rgb_array'],
				'video.frames_per_second' : 50
				}

	def __init__(self):

		self.goal_x = np.random.randint(640)
		self.goal_y = np.random.randint(480)
		self.mass_rand_person = 1
		self.mass_ctrl_person = 0.5
		self.radius_rand_person = 1
		self.radius_ctrl_person = 10
		self.tau = 0.02
		self.speed_rand_person = 1
		self.speedx_ctrl_person = 0.8
		self.speedy_ctrl_person = 0.8
		self.num_rand_person = 0
		self.action_space = spaces.Discrete(4) #forward, backward , left , right
		self.observation_space = spaces.Box(np.array([0,0]),np.array([640,480]))
		self.framecount = 0
		self.seed()
		self.viewer = None
		self.state = None
		self.steps_beyond_done = None
		self.threshold = 100
		self.timepenalty = 20
		
	def seed(self, seed=None):

		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self,action):
		self.framecount += 1
		#assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
		state = self.state
		x,y, goal_x , goal_y , dist_from_goal = state
		##speed_x = self.speedx_ctrl_person if action[0]==1 else -self.speedx_ctrl_person
		##speed_y = self.speedy_ctrl_person if action[1]==1 else -self.speedy_ctrl_person
		#print("ACTION", action)
		speed_x = self.speedx_ctrl_person*action[0]
		speed_y = self.speedy_ctrl_person*action[1]
		new_x = x+speed_x
		new_y = y+speed_y

		dist = math.pow(self.goal_x-new_x,2)+math.pow(self.goal_y-new_y,2)

		self.state = (new_x,new_y, self.goal_x , self.goal_y ,dist)

		done = dist < self.threshold

		done = bool(done)
		if not done:

			reward = 1000 - dist - self.framecount*self.timepenalty
		else:

			print("you have done it!!!!!")

		return np.array(self.state) , reward , done , {}

	def reset(self):

		self.goal_x = np.random.randint(640)
		self.goal_y = np.random.randint(480)
		dist = math.pow(self.goal_x-10,2)+math.pow(self.goal_y-10,2)
		self.state = (10 , 10 , self.goal_x , self.goal_y , dist)
		return np.array(self.state)
	

	def render(self, mode = 'human'):


		screen_width = 650
		screen_height = 500

		if self.viewer is None:

			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(screen_width, screen_height)
			ctrl_person = rendering.make_circle(self.radius_ctrl_person)
			self.prTrans = rendering.Transform()
			ctrl_person.add_attr(self.prTrans)
			self.viewer.add_geom(ctrl_person)
			goal = rendering.FilledPolygon([(5,5),(5,-5),(-5,5),(-5,-5)])
			self.goalobj = rendering.Transform() 
			goal.add_attr(self.goalobj)
			self.viewer.add_geom(goal)

		if self.state is None: return None

		x = self.state
		self.prTrans.set_translation(x[0],x[1])
		self.goalobj.set_translation(x[2],x[3])

		return self.viewer.render(return_rgb_array = mode == 'human')

	def close(self):
		if self.viewer: self.viewer.close()

