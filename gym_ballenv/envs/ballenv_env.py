import gym
import math
import time
import numpy as np
from gym import error ,spaces, utils
from gym.utils import seeding



_screen_height = 100
_screen_width = 100
_stripobsx = 0
_stripobsy = 20
_stripgoalx = 100
_stripgoaly = 20
_stripagentx = 100
_stripagenty = 10
class obstacles(object):
	"""docstring for obstacles"""
	goals = None
	def __init__(self):

		self.x = np.random.randint(_stripobsx,_screen_width-_stripobsx)
		self.y = np.random.randint(_stripobsy,_screen_height-_stripobsy)
		self.curr_goal = None #this is a tuple (x,y)
		self.speed = 0
		self.goal_change_counter = None
		self.curr_counter = 0
		self.proximity_threshold_1 = 0
		self.threshold_1_penalty = 0
		self.proximity_threshold_2 = 0 #threshold 2 is the tighter margin and thus carries a bigger penalty when violated
		self.threshold_2_penalty = 0

		

class BallEnv(gym.Env):

	metadata = {'render.modes': ['human' , 'rgb_array'],
				'video.frames_per_second' : 100
				}

	def __init__(self):

		self.goal_x = np.random.randint(_screen_width)
		self.goal_y = np.random.randint(_screen_height)
		self.mass_rand_person = 1
		self.mass_ctrl_person = 0.5
		self.radius_rand_person = 20
		self.radius_ctrl_person = 5
		self.tau = 0.02
		self.speed_rand_person = 1
		self.speedx_ctrl_person = 2
		self.speedy_ctrl_person = 2
		self.agent_uncertainty_threshold = None
		self.num_rand_person = 0
		self.action_space = spaces.Discrete(4) #forward, backward , left , right
		self.observation_space = spaces.Box(np.array([0,0]),np.array([_screen_width,_screen_height]))
		self.framecount = 0
		self.seed()
		self.viewer = None
		self.state = None
		self.steps_beyond_done = None
		self.threshold_goal = 10
		self.timepenalty = 0
		self.total_reward_accumulated = 0
		self.reward_threshold = -20000
		self.old_dist = None
		self.total_distance = None

		#information related to the obstacles
		self.obstacle_speed_list = []
		self.obstacle_goal_list = []
		self.no_of_static_obstacles = 0
		self.no_of_dynamic_obstacles = 0
		self.obs_uncertainity_threshold = None
		self.total_obstacles = None
		self.obstacle_list = []
		self.static_obstacle_list = []
		self.dynamic_obstacle_list = []
		self.obstacle_transform_list = []
		self.st_obs_prox_thresh = []
		self.dy_obs_prox_thresh = []
		self.st_obs_prox_penalty = []
		self.dy_obs_prox_penalty = []

	def customize_environment(self,args):
		#print('QQQQ',self.no_of_static_obstacles)
		#print(type(args))
		#print("ISIUDE")
		self.no_of_static_obstacles = args.static_obstacles
		self.no_of_dynamic_obstacles = args.dynamic_obstacles
		#obstacle_speed
		self.obstacle_speed_list = args.obstacle_speed
		self.goal_change_step  = args.time_step_for_change
		self.obs_uncertainity_threshold = args.rd_th_obs
		self.agent_uncertainty_threshold = args.rd_th_agent
		temp_goal_list =  []
		for tup in args.obs_goal_position:
			tuplist = tup.strip().split(',')
			x = int(tuplist[0])
			y = int(tuplist[1])
			temp_goal_list.append((x,y))
		self.obstacle_goal_list = temp_goal_list
		self.total_obstacles = self.no_of_static_obstacles+self.no_of_dynamic_obstacles
		self.st_obs_prox_thresh = args.static_thresholds
		self.dy_obs_prox_thresh = args.dynamic_thresholds
		self.st_obs_prox_penalty = args.static_penalty
		self.dy_obs_prox_penalty = args.dynamic_penalty
		#print('ZZZZ',self.no_of_static_obstacles)


	def reset(self):
		self.obstacle_list = []
		self.goal_x = np.random.randint(_screen_width-_stripgoalx,_screen_width)
		self.goal_y = np.random.randint(_screen_height-_stripgoaly,_screen_height)
		agent_x = np.random.randint(0,_stripagentx)
		agent_y = np.random.randint(0,_stripagenty)
		dist = math.sqrt(math.pow(self.goal_x-agent_x,2)+math.pow(self.goal_y-agent_y,2))
		self.old_dist = dist
		while(1):
			if self.calculate_distance((self.goal_x,self.goal_y),(agent_x,agent_y))<50:
				agent_x = np.random.randint(0,_stripagentx)
				agent_y = np.random.randint(0,_stripagenty)
			else:
				break
		self.state = [(agent_x , agent_y) , (self.goal_x , self.goal_y) , dist]
		#print('PPPP',self.total_obstacles)
		self.total_reward_accumulated = 0
		#intialize the static obstacles
		for i in range(self.no_of_static_obstacles):
			while(1):
				#print("hit")
				temp_obs = obstacles()
				temp_obs.proximity_threshold_1 = self.st_obs_prox_thresh[0]
				temp_obs.proximity_threshold_2 = self.st_obs_prox_thresh[1]
				temp_obs.threshold_1_penalty = self.st_obs_prox_penalty[0]
				temp_obs.threshold_2_penalty = self.st_obs_prox_penalty[1]
				#print("agent",self.state[0])
				#print("goal",self.state[1])
				#print("obstacle",temp_obs.x,temp_obs.y)
				#print("Agent check",self.check_overlap_rect((temp_obs.x,temp_obs.y),(agent_x,agent_y),self.radius_rand_person))
				#print("Goal check",self.check_overlap_rect((temp_obs.x,temp_obs.y),(self.goal_x,self.goal_y),self.radius_rand_person))
				#if (self.check_overlap((temp_obs.x,temp_obs.y),(agent_x,agent_y)) and self.check_overlap((temp_obs.x,temp_obs.y),(self.goal_x,self.goal_y))):
				if (not self.check_overlap_rect((temp_obs.x,temp_obs.y),(agent_x,agent_y),self.radius_rand_person) and not self.check_overlap_rect((temp_obs.x,temp_obs.y),(self.goal_x,self.goal_y),self.radius_rand_person)):
					self.static_obstacle_list.append(temp_obs)
					self.state.append((temp_obs.x,temp_obs.y))
					self.obstacle_list.append(temp_obs)
					break
		#print(len(self.obstacle_list))

		#initialize the dynamic obstacles
		for i in range(self.no_of_dynamic_obstacles):
			temp_obs = obstacles()
			temp_obs.proximity_threshold_1 = self.dy_obs_prox_thresh[0]
			temp_obs.proximity_threshold_2 = self.dy_obs_prox_thresh[1]
			temp_obs.threshold_1_penalty = self.dy_obs_prox_penalty[0]
			temp_obs.threshold_2_penalty = self.dy_obs_prox_penalty[1]
			temp_obs.speed = self.obstacle_speed_list[i]
			temp_obs.curr_goal = self.obstacle_goal_list[i]
			temp_obs.goal_change_counter = self.goal_change_step
			self.dynamic_obstacle_list.append(temp_obs)
			self.state.append((temp_obs.x,temp_obs.y))
			self.obstacle_list.append(temp_obs)

		self.total_distance = self.calculate_distance(self.state[0],self.state[1])
		return np.array(self.state)


	def get_accumulated_reward(self):

		return self.total_reward_accumulated

	def seed(self, seed=None):

		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def calculate_distance(self,tup1, tup2):

		x_diff = tup1[0] - tup2[0]
		y_diff = tup1[1] - tup2[1]
		return math.sqrt(math.pow(x_diff,2)+math.pow(y_diff,2))

	def check_overlap(self,tup1,tup2):

		dist = self.calculate_distance(tup1,tup2)
		if dist>(self.radius_rand_person+self.radius_ctrl_person):
			return False
		else:
			return True

	def check_overlap_rect(self,tup1,tup2,rad):
		if abs(tup1[0]-tup2[0])<(rad+self.radius_ctrl_person) and abs(tup1[1]-tup2[1])<(rad/2+self.radius_ctrl_person):
			return True
		else:
			return False


	def calculate_reward(self):

		#reward = 0
		done = False
		dist = self.calculate_distance(self.state[0],self.state[1])
		reward = -self.timepenalty
		reward += (self.old_dist-dist)/self.total_distance

		for obs in self.obstacle_list:
			'''
			temp_dist = self.calculate_distance(self.state[0] , (obs.x,obs.y))
			if temp_dist <= obs.proximity_threshold_2:
				reward -= obs.threshold_2_penalty
				done = True
				continue
			if temp_dist <= obs.proximity_threshold_1:
				reward -= obs.threshold_1_penalty
				done = True
				continue
			'''
			done = self.check_overlap(self.state[0],(obs.x,obs.y))
			#done = self.check_overlap_rect(self.state[0],(obs.x,obs.y),self.radius_rand_person)
			if done:
				reward -= obs.threshold_2_penalty
				break
		if dist < self.threshold_goal:

			reward += 1

		return reward,done


	def step(self,action):
		self.framecount += 1
		#assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
		state = self.state
		self.old_dist = state[2]
		#print(len(self.obstacle_list))
		#print(state)
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
		if new_x<0:
			new_x = 0
		if new_y<0:
			new_y = 0
		if new_x>_screen_width:
			new_x = _screen_width
		if new_y>_screen_height:
			new_y = _screen_height

		##loop to move each of the dynamic obstacles using move function
		for i in self.dynamic_obstacle_list:

			self.move_obstacles(i)
		##
		#print("old_dist",self.old_dist)

		dist = math.sqrt(math.pow(self.goal_x-new_x,2)+math.pow(self.goal_y-new_y,2))
		#print("new_dist",dist)
		self.state = None
		self.state = [(new_x,new_y), (self.goal_x , self.goal_y) ,dist]

		for i in self.obstacle_list:
			self.state.append((i.x,i.y))

		goal_flag = dist < self.threshold_goal	

		reward,obs_flag = self.calculate_reward()

		self.total_reward_accumulated += reward

		#if self.total_reward_accumulated <= self.reward_threshold:

		#	penalty_flag = True

		done = goal_flag or obs_flag
		#print("Accumulated_reward :", self.total_reward_accumulated)
		#time.sleep(.2)
		return np.array(self.state) , reward , done , {}
		#return 


	

	def render_obstacle(self, obstacle):
		from gym.envs.classic_control import rendering
		l = self.radius_rand_person
		if obstacle.speed==0:
			rend_obs = rendering.make_circle(self.radius_rand_person)
			#rend_obs = rendering.FilledPolygon([(-l,-l/2),(-l,l/2),(l,l/2),(l,-l/2)])
			rend_obs.set_color(100,0,0)
			#rendering._add_attrs(rend_obs,rendering.Color((100,100,200,10)))
			#rend_obs.add_attrs(obs_color)
		else:
			rend_obs = rendering.make_circle(self.radius_rand_person)
			rend_obs.set_color(0,100,0)
			#obs_color = rendering.Color((0,1,0,0))
			#rend_obs.add_attrs(obs_color)

		obs = rendering.Transform()
		#obs_color = rendering.Color((1,0,0))
		rend_obs.add_attr(obs)
		self.viewer.add_geom(rend_obs)
		self.obstacle_transform_list.append(obs)



	def place_obstacle(self, obstacle , obs_transform):
		obs_transform.set_translation(obstacle.x,obstacle.y)



	def move_obstacles(self, obstacle):
		move_list = [(1,1) , (1,-1) , (1, 0) , (0,1) , (0,-1),(0,0) , (-1,1),(-1,-1),(-1,-1)]
		if obstacle.speed!=None:

			if obstacle.curr_counter< obstacle.goal_change_counter:

				tempx = obstacle.curr_goal[0] - obstacle.x
				tempy = obstacle.curr_goal[1] - obstacle.y
				if tempx!=0 and tempy!=0:
					if np.random.randint(100)<self.obs_uncertainity_threshold:

						obstacle.x += (tempx/abs(tempx))*obstacle.speed
						obstacle.y += (tempy/abs(tempy))*obstacle.speed

				
					else:

						i = np.random.randint(9)
						obstacle.x += move_list[i][0]*obstacle.speed
						obstacle.y += move_list[i][1]*obstacle.speed
				else:

					i = np.random.randint(9)
					obstacle.x += move_list[i][0]*obstacle.speed
					obstacle.y += move_list[i][1]*obstacle.speed
				obstacle.curr_counter+=1
			else:

				newGoalList = [ i for i in self.obstacle_goal_list if i != obstacle.curr_goal]
				obstacle.curr_goal = newGoalList[np.random.randint(len(newGoalList))]
				obstacle.curr_counter = 0
		


	def render(self, mode = 'rgb_array' ,close = True):


		screen_width = _screen_width
		screen_height = _screen_height

		if self.viewer is None:

			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(screen_width, screen_height,display=None)
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

		return self.viewer.render(return_rgb_array = mode == 'rgb_array')

	def close(self):
		if self.viewer: self.viewer.close()

