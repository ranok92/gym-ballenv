import pygame
import numpy as np
import math
import gym 


#random.seed(4)
_screen_height = 500
_screen_width = 500
_stripobsx = 0
_stripobsy = 20
_stripgoalx = 500
_stripgoaly = 20
_stripagentx = 500
_stripagenty = 10
_max_agent_speed = 5



class Obstacle():
	goals = None
	def __init__(self):

		self.x = np.random.randint(_stripobsx,_screen_width-_stripobsx)
		self.y = np.random.randint(_stripobsy,_screen_height-_stripobsy)
		self.rad = 10
		self.curr_goal = None #this is a tuple (x,y)
		self.speed = 0
		self.goal_change_counter = None
		self.curr_counter = 0


class createBoard():

	def __init__(self,height = _screen_height , width = _screen_width , agent_radius = 10 , static_obstacles = 10 , dynamic_obstacles = 0 , static_obstacle_radius = 10 , dynamic_obstacle_radius = 0 , obstacle_speed_list = []):
		print pygame.init()
		self.clock = pygame.time.Clock()


		self.display = True
		self.gameExit = False
		self.height = height
		self.width = width
		self.agent_radius = agent_radius


		self.no_static_obstacles = static_obstacles
		self.no_dynamic_obstacles = dynamic_obstacles
		self.total_obs = self.no_static_obstacles+self.no_dynamic_obstacles
		self.rad_static_obstacles = static_obstacle_radius
		self.rad_dynamic_obstacles = dynamic_obstacle_radius
		self.agent_action_flag = False

		self.agent_x = None 
		self.agent_y = None
		self.goal_x = None
		self.goal_y = None
		self.old_dist = None
		self.goal_threshold = 5
		self.total_distance = None
		self.obstacle_speed_list = obstacle_speed_list
		self.static_obstacle_list = []
		self.dynamic_obstacle_list = []
		self.obstacle_list = []

		self.state = None
		self.reward = None
		self.total_reward_accumulated = None

		self.white = (255,255,255)
		self.red = (255,0,0)
		self.green = (0,255,0)
		self.blue = (0,0,255)
		self.black = (0,0,0)
		self.caption = 'social navigation world'
		



	def calculate_distance(self,tup1, tup2):

		x_diff = tup1[0] - tup2[0]
		y_diff = tup1[1] - tup2[1]
		return math.sqrt(math.pow(x_diff,2)+math.pow(y_diff,2))

	def check_overlap(self,tup1,tup2):

		dist = self.calculate_distance(tup1,tup2)
		if dist>(self.rad_static_obstacles+self.agent_radius):
			return False
		else:
			return True

	def take_action_from_user(self):
		(a,b,c) = pygame.mouse.get_pressed()
		
		x = 0.0001
		y = 0.0001
		for event in pygame.event.get():
			if event.type==pygame.MOUSEBUTTONDOWN:
				self.agent_action_flag=True
			if event.type==pygame.MOUSEBUTTONUP:
				self.agent_action_flag=False
		if self.agent_action_flag:	
			(x,y) = pygame.mouse.get_pos()
			x = x - self.state[0][0]
			y = y - self.state[0][1]
			if np.hypot(x,y)>_max_agent_speed:
				normalizer = _max_agent_speed/(np.hypot(x,y))
			#print x,y
			else:
				normalizer = 1
			return (x*normalizer,y*normalizer)

		return (0,0)



	def check_overlap_rect(self,tup1,tup2,rad):
		if abs(tup1[0]-tup2[0])<(rad+self.agent_radius) and abs(tup1[1]-tup2[1])<(rad/2+self.rad_static_obstacles):
			return True
		else:
			return False

	def generate_randomval(self,lower,upper):

		i = np.random.ranf()
		return lower+i*(upper-lower)


	def reset(self):
		self.agent_action_flag = False
		if self.display:
			self.gameDisplay = pygame.display.set_mode((self.width ,self.height))
			pygame.display.set_caption('social navigation world')
			self.gameDisplay.fill(self.white)

		self.obstacle_list = []
		self.goal_x = self.generate_randomval(_screen_width-_stripgoalx,_screen_width)
		self.goal_y = self.generate_randomval(_screen_height-_stripgoaly,_screen_height)
		agent_x = self.generate_randomval(0,_stripagentx)
		agent_y = self.generate_randomval(0,_stripagenty)
		dist = math.sqrt(math.pow(self.goal_x-agent_x,2)+math.pow(self.goal_y-agent_y,2))
		self.old_dist = dist
		while(1):
			if self.calculate_distance((self.goal_x,self.goal_y),(agent_x,agent_y))<50:
				agent_x = self.generate_randomval(0,_stripagentx)
				agent_y = self.generate_randomval(0,_stripagenty)
			else:
				break
		self.state = [(agent_x , agent_y) , (self.goal_x , self.goal_y) , dist]
		#print('PPPP',self.total_obstacles)
		self.total_reward_accumulated = 0
		#intialize the static obstacles
		for i in range(self.no_static_obstacles):
			while(1):
				#print("hit")
				temp_obs = Obstacle()
				if (not self.check_overlap_rect((temp_obs.x,temp_obs.y),(agent_x,agent_y),self.rad_static_obstacles) and not self.check_overlap_rect((temp_obs.x,temp_obs.y),(self.goal_x,self.goal_y),self.rad_static_obstacles)):
					self.static_obstacle_list.append(temp_obs)
					self.state.append((temp_obs.x,temp_obs.y))
					self.obstacle_list.append(temp_obs)
					break
		#print(len(self.obstacle_list))

		#initialize the dynamic obstacles
		for i in range(self.no_dynamic_obstacles):
			temp_obs = obstacles()
			temp_obs.speed = self.obstacle_speed_list[i]
			temp_obs.curr_goal = self.obstacle_goal_list[i]
			temp_obs.goal_change_counter = self.goal_change_step
			self.dynamic_obstacle_list.append(temp_obs)
			self.state.append((temp_obs.x,temp_obs.y))
			self.obstacle_list.append(temp_obs)

		self.total_distance = self.calculate_distance(self.state[0],self.state[1])
		
		return np.array(self.state)


	def renderObstacle(self,obs):
		pygame.draw.circle(self.gameDisplay , self.red , (obs.x , obs.y) , obs.rad)

	def render(self): #renders the screen using the current state of the environment

		self.gameDisplay.fill(self.white)
		for obs in self.obstacle_list:
			self.renderObstacle(obs)
		#draw goal
		pygame.draw.rect(self.gameDisplay , self.black , [self.goal_x,self.goal_y,10,10])
		#draw agent
		pygame.draw.circle(self.gameDisplay , self.black , (int(self.state[0][0]) , int(self.state[0][1])),self.agent_radius)
		pygame.display.update()
		self.clock.tick(30)
	#updates the position of the objects in the environment according to the dynamics
	#action is a tuple (x,y), which gets added to the current 
	#position of the agent

	def step(self,action): 

		self.old_dist = self.calculate_distance(self.state[0], self.state[1])

		newx = self.state[0][0]+action[0]
		newy = self.state[0][1]+action[1]

		if newx < 0:
			newx = 0
		if newx > _screen_width:
			newx = _screen_width
		if newy < 0:
			newy = 0
		if newy > _screen_height:
			newy = _screen_height
		self.state[0] = (newx , newy)

		reward, done = self.calc_reward()
		#print self.state[0]
		return np.array(self.state) , reward, done , {}

# -1 if it hits an obstacle or fraction of the distance it travels towards
#the goal with a total of 1 when it reaches the goal

	def calc_reward(self):

		done = False
		for obs in self.obstacle_list:

			done = self.check_overlap(self.state[0] , (obs.x,obs.y))
			if done:
				return -1 , done

		if self.calculate_distance(self.state[0],self.state[1]) < self.goal_threshold:

			done = True
			reward = 1-self.total_reward_accumulated
			self.total_reward_accumulated+= reward

			return reward, done

		else:
			cur_dist = self.calculate_distance(self.state[0] , self.state[1])

			diff_dist = self.old_dist - cur_dist
			reward = diff_dist/self.total_distance
			self.total_reward_accumulated+= reward
			return reward,done



cb = createBoard()

for i in range(1):
	print "Here"
	cb.reset()
	for j in range(300000):
		if cb.display:
			cb.render()
		action = cb.take_action_from_user()
		print action
		state ,reward ,done , _ = cb.step(action)
		if done:
			break
		#print reward 
		#print cb.total_reward_accumulated
		
	pygame.quit()

