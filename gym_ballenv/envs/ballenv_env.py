import gym
import math
import numpy as np
from gym import error ,spaces, utils
from gym.utils import seeding

class obstacles(object):
	"""docstring for obstacles"""
	goals = None
	def __init__(self):

		self.x = np.random.randint(640)
		self.y = np.random.randint(480)
		self.curr_goal = (300,200) #this is a tuple (x,y)
		self.speed = 10
		self.proximity_threshold_1 = 1000
		self.threshold_1_penalty = -100000
		self.proximity_threshold_2 = 90
		self.threshold_2_penalty = -300000

		

class BallEnv(gym.Env):

	metadata = {'render.modes': ['human' , 'rgb_array'],
				'video.frames_per_second' : 50
				}

	def __init__(self):

		self.goal_x = np.random.randint(640)
		self.goal_y = np.random.randint(480)
		self.mass_rand_person = 1
		self.mass_ctrl_person = 0.5
		self.radius_rand_person = 7
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

		self.no_of_static_obstacles = 6
		self.no_of_dynamic_obstacles = 0
		self.total_obstacles = self.no_of_static_obstacles+self.no_of_dynamic_obstacles
		self.obstacle_list = []
		self.obstacle_transform_list = []


	def seed(self, seed=None):

		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def calculate_distance(self,tup1, tup2):

		x_diff = tup1[0] - tup2[0]
		y_diff = tup1[1] - tup2[1]
		return math.pow(x_diff,2)+math.pow(y_diff,2)


	def calculate_reward(self):

		dist = self.calculate_distance(self.state[0],self.state[1])
		reward = 1000 - dist - self.framecount*self.timepenalty

		for obs in self.obstacle_list:

			temp_dist = self.calculate_distance(self.state[0] , (obs.x,obs.y))
			if temp_dist <= obs.proximity_threshold_2:
				reward += obs.threshold_2_penalty
				continue
			if temp_dist <= obs.proximity_threshold_1:
				reward += obs.threshold_1_penalty
				continue

		return reward


	def step(self,action):
		self.framecount += 1
		#assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
		state = self.state
		print(len(self.obstacle_list))
		print(state)
		cur_coord = state[0]
		goal_coord = state[1]
		dist_from_goal = state[2]
		x = cur_coord[0]
		y = cur_coord[1]
		##speed_x = self.speedx_ctrl_person if action[0]==1 else -self.speedx_ctrl_person
		##speed_y = self.speedy_ctrl_person if action[1]==1 else -self.speedy_ctrl_person
		#print("ACTION", action)
		speed_x = self.speedx_ctrl_person*action[0]
		speed_y = self.speedy_ctrl_person*action[1]
		new_x = x+speed_x
		new_y = y+speed_y
		##

		##loop to move each of the dynamic obstacles using move function
		for i in self.obstacle_list:

			self.move_obstacles(i)
		##
		dist = math.pow(self.goal_x-new_x,2)+math.pow(self.goal_y-new_y,2)
		self.state = None
		self.state = [(new_x,new_y), (self.goal_x , self.goal_y) ,dist]

		for i in self.obstacle_list:
			self.state.append((i.x,i.y))

		done = dist < self.threshold

		done = bool(done)
		if not done:

			reward = self.calculate_reward()
		else:

			print("you have done it!!!!!")

		return np.array(self.state) , reward , done , {}

	def reset(self):
		self.obstacle_list = []
		self.goal_x = np.random.randint(640)
		self.goal_y = np.random.randint(480)
		dist = math.pow(self.goal_x-10,2)+math.pow(self.goal_y-10,2)
		self.state = [(10 , 10) , (self.goal_x , self.goal_y) , dist]
		for i in range(self.total_obstacles):
			temp_obs = obstacles()
			self.obstacle_list.append(temp_obs)
			self.state.append((temp_obs.x,temp_obs.y))
		print(len(self.obstacle_list))
		return np.array(self.state)
	

	def render_obstacle(self, obstacle):
		from gym.envs.classic_control import rendering
		rend_obs = rendering.make_circle(self.radius_rand_person)
		obs = rendering.Transform()
		rend_obs.add_attr(obs)
		self.viewer.add_geom(rend_obs)
		self.obstacle_transform_list.append(obs)

	def place_obstacle(self, obstacle , obs_transform):

		obs_transform.set_translation(obstacle.x,obstacle.y)



	def move_obstacles(self, obstacle):
		move_list = [(1,1) , (1,-1) , (1, 0) , (0,1) , (0,-1),(0,0) , (-1,1),(-1,-1),(-1,-1)]
		if obstacle.speed!=None:

			tempx = obstacle.curr_goal[0] - obstacle.x
			tempy = obstacle.curr_goal[1] - obstacle.y

			if np.random.randint(100)<68:

				obstacle.x += (tempx/abs(tempx))*obstacle.speed
				obstacle.y += (tempy/abs(tempy))*obstacle.speed

				
			else:

				i = np.random.randint(9)
				obstacle.x += move_list[i][0]*obstacle.speed
				obstacle.y += move_list[i][1]*obstacle.speed
				
				
		


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
			for obs in self.obstacle_list:
				self.render_obstacle(obs)

		if self.state is None: return None

		x = self.state
		self.prTrans.set_translation(x[0][0],x[0][1])
		self.goalobj.set_translation(x[1][0],x[1][1])
		for i in range(self.total_obstacles):
			self.place_obstacle(self.obstacle_list[i],self.obstacle_transform_list[i])

		return self.viewer.render(return_rgb_array = mode == 'human')

	def close(self):
		if self.viewer: self.viewer.close()

