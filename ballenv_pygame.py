import pygame
import numpy as np
import math
#import gym 
import os
import featureExtractor
#random.seed(4)
_screen_height = 100
_screen_width = 100
_stripobsx = 0
_stripobsy = 0
_stripgoalx = 100 #the closer this to screen_height the more the randomness
_stripgoaly = 100 #the closer this to screen_height the more the randomness
_stripagentx = 100
_stripagenty = 100
_max_agent_speed = 5




class Obstacle:
    goals = None

    def __init__(self,id,xpos = None , ypos = None , xvel = None , yvel = None, radius = None):
        self.id = id
        if xpos==None:
            self.x = np.random.randint(_stripobsx,_screen_width-_stripobsx)
        else:
            self.x = xpos

        if ypos==None:
            self.y = np.random.randint(_stripobsy,_screen_height-_stripobsy)
        else:
            self.y = ypos
        if radius==None:
            self.rad = 20
        else:
            self.rad = radius
        self.curr_goal = None #this is a tuple (x,y)
        if xvel == None:
            self.vel_x = 0
        else:
            self.vel_x = xvel
        if yvel == None:
            self.vel_y = 0
        else:
            self.vel_y = yvel

        self.goal_change_counter = None
        self.curr_counter = 0





#this class is created to feed information from a trajectory file frame by frame
#and simulate that in the pygame environment.

class createBoardperFrame():

	def __init__(self,annotation_file):
		
		# general variables
		self.agent_x = None
		self.agent_y = None
		self.goal = [0,0]
		self.state = None
		self.reward = None
		self.total_reward_accumulated = 0
		self.old_dist = None
		self.total_distance = None
		self.goal_threshold = None
		self.agent_radius = 25
		self.agent_action_flag = True
		# variables for annotation method
		self.annotation_dict = {}
		self.color_dict = {'"Biker"':(0,0,255), '"Pedestrian"' : (0,255,0), '"Car"' : (255,255,0) , '"Bus"' : (0,0,0) , '"Skater"' : (100,100,255) , '"Cart"':(255,255,255)}
		self.frame_count = 0
		self.finalFrameCount = -1
		self.current_Frame_info = None
		self.display = True
		# variables for pygame implementation
		self.gameDisplay = None
		self.done = False
		self.clock = pygame.time.Clock()
		self.width = _screen_width
		self.height = _screen_height
		self.white = (255,255,255)
		annotation_list = []
		print(pygame.init())
		if not os.path.isfile(annotation_file):
			print ("The annotation file does not exist.")
			return 0
		with open(annotation_file) as f:
			for line in f:
				line = line.strip().split(' ')
				annotation_list.append(line)
	
		#this part is hardcoded for the drone dataset
		#for a different dataset change the code below to create 
		#the dictionary accordingly.
		#dictionary key : frame number	
		#for each key there is a list, the list is a 2d list where each element
		#
		for entry in annotation_list:
			if entry[5] not in self.annotation_dict:
				self.annotation_dict[entry[5]] = []
				if self.finalFrameCount < int(entry[5]):
					self.finalFrameCount = int(entry[5])
			self.annotation_dict[entry[5]].append(entry)

	def reset(self):


		self.goal[0] = self.generate_randomval(_screen_width-_stripgoalx,_screen_width)
		self.goal[1] = self.generate_randomval(_screen_height-_stripgoaly,_screen_height)
		agent_x = self.generate_randomval(0,_stripagentx)
		agent_y = self.generate_randomval(0,_stripagenty)
		dist = math.sqrt(math.pow(self.goal[0]-agent_x,2)+math.pow(self.goal[1]-agent_y,2))
		self.old_dist = dist
		self.total_distance = dist
		while(1):
			if self.calculate_distance((self.goal[0],self.goal[1]),(agent_x,agent_y))<50:
				agent_x = self.generate_randomval(0,_stripagentx)
				agent_y = self.generate_randomval(0,_stripagenty)
			else:
				break
		#self.state = [(agent_x , agent_y) , (self.goal_x , self.goal_y) , dist]
		self.agent_x = agent_x
		self.agent_y = agent_y

		if self.display:
			self.gameDisplay = pygame.display.set_mode((self.width ,self.height))
			pygame.display.set_caption('social navigation world')
			self.gameDisplay.fill(self.white)

	def step(self,action): 

		self.old_dist = self.calculate_distance(self.agent, self.goal)

		newx = self.agent_x+action[0]
		newy = self.agent_y+action[1]

		if newx < 0:
			newx = 0
		if newx > _screen_width:
			newx = _screen_width
		if newy < 0:
			newy = 0
		if newy > _screen_height:
			newy = _screen_height
		self.agent_x = newx 
		self.agent_y = newy

		reward, done = self.calc_reward()
		#print self.state[0]
		return reward, done , {}


	def render(self):

		self.gameDisplay.fill(self.white)

		if self.frame_count <= self.finalFrameCount:
			self.current_Frame_info = self.annotation_dict[str(self.frame_count)]
			obstacle_list = self.annotation_dict[str(self.frame_count)]
			self.place_obstacles(obstacle_list)
			pygame.draw.circle(self.gameDisplay , (65,122,45), (int(self.agent[0]) , int(self.agent[1])),self.agent_radius)
			pygame.display.update()
			self.clock.tick(30)
			self.frame_count += 1
		else:
			self.done = True


	def calc_reward(self):

		done = False
		for obs in self.current_Frame_info:

			done = self.check_overlap_obstacle(obs)
			if done:
				return -1 , done

		if self.calculate_distance(self.agent,self.goal) < self.goal_threshold:

			done = True
			reward = 1-self.total_reward_accumulated
			self.total_reward_accumulated+= reward

			return reward, done

		else:
			cur_dist = self.calculate_distance(self.agent , self.goal)

			diff_dist = self.old_dist - cur_dist
			reward = diff_dist/self.total_distance
			self.total_reward_accumulated+= reward
			return reward,done

	def check_overlap_obstacle(self,obstacle_info):

		xmin = int(obstacle_info[1])
		ymin = int(obstacle_info[2])
		xmax = int(obstacle_info[3])
		ymax = int(obstacle_info[4])
		overlap = False
		#topleft corner
		if xmin-self.agent_radius < self.agent[0] <xmax+ self.agent_radius and ymin-self.agent_radius <self.agent[1] < ymax +self.agent_radius:

			overlap = True
		return overlap


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
			x = x - self.agent[0]
			y = y - self.agent[1]
			if np.hypot(x,y)>_max_agent_speed:
				normalizer = _max_agent_speed/(np.hypot(x,y))
			#print x,y
			else:
				normalizer = 1
			return (x*normalizer,y*normalizer)

		return (0,0)



	#a simplified version where there are just 4actions
	def take_action_from_userKeyboard(self):

		while (True):
			for event in pygame.event.get():
				if event.type == pygame.KEYDOWN:
					key = pygame.key.get_pressed()
					if key[pygame.K_UP]:
						self.agent_action_keyboard[0]=True
					if key[pygame.K_RIGHT]:
						self.agent_action_keyboard[1]=True
					if key[pygame.K_LEFT]:
						self.agent_action_keyboard[3]=True
					if key[pygame.K_DOWN]:
						self.agent_action_keyboard[2]=True

				if event.type == pygame.KEYUP:

					if event.key == pygame.K_UP:
						self.agent_action_keyboard[0]=False
					if event.key == pygame.K_RIGHT:
						self.agent_action_keyboard[1]=False
					if event.key == pygame.K_LEFT:
						self.agent_action_keyboard[3]=False
					if event.key == pygame.K_DOWN:
						self.agent_action_keyboard[2]=False

			for i in range(len(self.agent_action_keyboard)):
				if self.agent_action_keyboard[i]==True:
					return i

		return None







	def generate_randomval(self,lower,upper):

		i = np.random.ranf()
		return lower+i*(upper-lower)






	def calculate_distance(self,tup1, tup2):

		x_diff = tup1[0] - tup2[0]
		y_diff = tup1[1] - tup2[1]
		return math.sqrt(math.pow(x_diff,2)+math.pow(y_diff,2))


	def place_obstacles(self, frame_info):

		for element in frame_info:
			if element[6] != 1:
				left = int(element[1])
				top = int(element[2])
				width = int(element[3]) - left
				height = int(element[4]) - top
				rect_i = pygame.Rect(left,top,width,height)
				pygame.draw.rect(self.gameDisplay , self.color_dict[element[9]] , rect_i , 0)

				#cv2.rectangle(image , (int(element[1]),int(element[2])),(int(element[3]),int(element[4])) , color_dict[element[9]] , 2)




#this is the class to be used for a regular game environment

class createBoard():

	def __init__(self,height = _screen_height , display = False, width = _screen_width , agent_radius = 10 , static_obstacles = 0 , dynamic_obstacles = 0 , static_obstacle_radius = 10 , dynamic_obstacle_radius = 0 , obstacle_speed_list = []):
		print(pygame.init())
		print("here")
		self.clock = pygame.time.Clock()


		self.display = display
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
		self.agent_x_vel = 0
		self.agent_y_vel = 0
		self.agentStart = None #this is only for the DQN method so that the agent 
								#starts from the same position in each of the episodes
		self.goal_x = None
		self.goal_y = None
		self.old_dist = None
		self.goal_threshold = 15
		self.total_distance = None
		self.obstacle_speed_list = obstacle_speed_list
		self.static_obstacle_list = []
		self.dynamic_obstacle_list = []
		self.obstacle_list = []
		self.agent_action_keyboard = [False for i in range(4)]

		#state : list
		#state[0] - tuple containing agent current position
		#state[1] - tuple containing goal position.
		#state[2] - distance from goal
		#state[3 - end] - tuple obstacle position


		self.state = None
		self.sensor_readings = None
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

	def check_overlap(self,tup1,tup2,thresh = 0):

		dist = self.calculate_distance(tup1,tup2) - thresh
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
			x = x - self.agent_x
			y = y - self.agent_y
			if np.hypot(x,y)>_max_agent_speed:
				normalizer = _max_agent_speed/(np.hypot(x,y))
			#print x,y
			else:
				normalizer = 1
			return (x*normalizer,y*normalizer)

		return (0,0)



	def take_action_from_userKeyboard(self):

		while (True):
			for event in pygame.event.get():
				if event.type == pygame.KEYDOWN:
					key = pygame.key.get_pressed()
					if key[pygame.K_UP]:
						self.agent_action_keyboard[0]=True
					if key[pygame.K_RIGHT]:
						self.agent_action_keyboard[1]=True
					if key[pygame.K_LEFT]:
						self.agent_action_keyboard[3]=True
					if key[pygame.K_DOWN]:
						self.agent_action_keyboard[2]=True

				if event.type == pygame.KEYUP:

					if event.key == pygame.K_UP:
						self.agent_action_keyboard[0]=False
					if event.key == pygame.K_RIGHT:
						self.agent_action_keyboard[1]=False
					if event.key == pygame.K_LEFT:
						self.agent_action_keyboard[3]=False
					if event.key == pygame.K_DOWN:
						self.agent_action_keyboard[2]=False

			for i in range(len(self.agent_action_keyboard)):
				if self.agent_action_keyboard[i]==True:
					return i

		return None



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
		#self.goal_x = 145
		#self.goal_y = 120
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
		self.agent_x = agent_x
		self.agent_y = agent_y
		self.agentStart = (agent_x,agent_y)
		self.total_reward_accumulated = 0
		#intialize the static obstacles
		for i in range(self.no_static_obstacles):
			while(1):
				#print("hit")
				temp_obs = Obstacle(self.rad_static_obstacles)
				#if (not self.check_overlap_rect((temp_obs.x,temp_obs.y),(agent_x,agent_y),self.rad_static_obstacles) and not self.check_overlap_rect((temp_obs.x,temp_obs.y),(self.goal_x,self.goal_y),self.rad_static_obstacles)):
				if (not self.check_overlap((temp_obs.x , temp_obs.y), (agent_x,agent_y) , thresh = 15)) and (not self.check_overlap((temp_obs.x , temp_obs.y), (self.goal_x,self.goal_y), thresh = 5)):
					self.static_obstacle_list.append(temp_obs)
					self.state.append((temp_obs.x,temp_obs.y ,temp_obs.rad))
					self.obstacle_list.append(temp_obs)
					break
		#print(len(self.obstacle_list))

		#initialize the dynamic obstacles
		for i in range(self.no_dynamic_obstacles):
			temp_obs = obstacles(self.rad_dynamic_obstacles)
			temp_obs.speed = self.obstacle_speed_list[i]
			temp_obs.curr_goal = self.obstacle_goal_list[i]
			temp_obs.goal_change_counter = self.goal_change_step
			self.dynamic_obstacle_list.append(temp_obs)
			self.state.append((temp_obs.x,temp_obs.y,temp_obs.rad))
			self.obstacle_list.append(temp_obs)

		self.total_distance = self.calculate_distance(self.state[0],self.state[1])
		self.sensor_readings = featureExtractor.featureExtractor(self.state,self.obstacle_list,(self.agent_x_vel,self.agent_y_vel),self.agent_radius)
		return np.array(self.state)

	#*** added new for convenience
	#this block is used to calculate the local window state representation
	#********************************************************************
	# def block_to_arrpos(self,window_size,x,y):
	#
	# 	a = (window_size**2-1)/2
	# 	b = window_size
	# 	pos = a+(b*y)+x
	#
	# 	return int(pos)
	#
	# #unlike the method in ActorCritic, this method just returns the state in the form of a numpy array
	# def get_state_BallEnv(self, window_size = 5):
	#
	# #state is a list of info where 1st position holds the position of the
	# #agent, 2nd the position of the goal , 3rd the distance after that,
	# #the positions of the obstacles in the world
	# 	#print(state)
	# 	window_size = window_size
	# 	block_width = 2
	#
	# 	window_rows = window_size
	# 	row_start =  (window_rows-1)/2
	# 	window_cols = window_size
	# 	col_start = (window_cols-1)/2
	#
	# 	ref_state = np.zeros(4+window_size**2)
	# 	#print(ref_state.shape)
	# 	a = (window_size**2-1)/2
	# 	ref_state[a+4] = 1
	# 	agent_pos = self.state[0]
	# 	goal_pos = self.state[1]
	# 	diff_x = goal_pos[0] - agent_pos[0]
	# 	diff_y = goal_pos[1] - agent_pos[1]
	# 	if diff_x >= 0 and diff_y >= 0:
	# 		ref_state[1] = 1
	# 	elif diff_x < 0  and diff_y >= 0:
	# 		ref_state[0] = 1
	# 	elif diff_x < 0 and diff_y < 0:
	# 		ref_state[3] = 1
	# 	else:
	# 		ref_state[2] = 1
	#
	# 	for i in range(3,len(self.state)):
	#
	# 		#as of now this just measures the distance from the center of the obstacle
	# 		#this distance has to be measured from the circumferance of the obstacle
	#
	# 		#new method, simulate overlap for each of the neighbouring places
	# 		#for each of the obstacles
	# 		obs_pos = self.state[i][0:2]
	# 		obs_rad = self.state[i][2]
	# 		for r in range(-row_start,row_start+1,1):
	# 			for c in range(-col_start,col_start+1,1):
	# 				#c = x and r = y
	# 				temp_pos = (agent_pos[0] + c*block_width , agent_pos[1] + r*block_width)
	# 				if self.checkOverlap(temp_pos,self.agentRad, obs_pos, obs_rad):
	# 					pos = self.block_to_arrpos(window_size,r,c)
	#
	# 					ref_state[pos]=1
	#
	# 	#state is as follows:
	# 		#first - tuple agent position
	# 		#second -
	# 	#ref_state = torch.from_numpy(ref_state).to(self.device)
	# 	#ref_state = ref_state.type(torch.cuda.FloatTensor)
	# 	#ref_state = ref_state.unsqueeze(0)
	#
	# 	return ref_state

	#******************************************************************************************



	def resetFixedstate(self):
		self.agent_action_flag = False
		if self.display:
			self.gameDisplay = pygame.display.set_mode((self.width ,self.height))
			pygame.display.set_caption('social navigation world')
			self.gameDisplay.fill(self.white)
		self.goal_x = 145
		self.goal_y = 120

		while True:
			agent_x = self.generate_randomval(0,_stripagentx)
			agent_y = self.generate_randomval(0,_stripagenty)
			#agent_x  = self.agentStart[0]
			#agent_y = self.agentStart[1]
			dist = math.sqrt(math.pow(self.goal_x-agent_x,2)+math.pow(self.goal_y-agent_y,2))
			self.old_dist = dist
			while(1):
				if self.calculate_distance((self.goal_x,self.goal_y),(agent_x,agent_y))<50:
					agent_x = self.generate_randomval(0,_stripagentx)
					agent_y = self.generate_randomval(0,_stripagenty)
				else:
					break
			self.state[0:3] = [(agent_x , agent_y) , (self.goal_x , self.goal_y) , dist]
			_,done = self.calc_reward()
			if not done:
				break



		#print('PPPP',self.total_obstacles)
		self.agent_x = agent_x
		self.agent_y = agent_y
		self.total_reward_accumulated = 0
		self.total_distance = self.calculate_distance(self.state[0],self.state[1])
		self.sensor_readings = featureExtractor.featureExtractor(self.state,self.obstacle_list,(self.agent_x_vel,self.agent_y_vel),self.agent_radius)
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
		
		
		self.state[2] = self.calculate_distance(self.state[0],self.state[1])
		self.agent_x = self.state[0][0]
		self.agent_y = self.state[0][1]
		reward, done = self.calc_reward()
		#print self.state[0]
		#print self.state
		self.sensor_readings = featureExtractor.featureExtractor(self.state,self.obstacle_list,(self.agent_x_vel,self.agent_y_vel),self.agent_radius)
		return np.asarray(self.state) , reward, done , {}

# -1 if it hits an obstacle or fraction of the distance it travels towards
#the goal with a total of 1 when it reaches the goal

	def calc_reward(self):
		#print 'The distance ',self.calculate_distance(self.state[0],self.state[1])
		done = False
		for obs in self.obstacle_list:

			done = self.check_overlap(self.state[0] , (obs.x,obs.y))
			if done:
				self.total_reward_accumulated+=-1
				return -1 , done

		if self.calculate_distance(self.state[0],self.state[1]) < self.goal_threshold:

			done = True
			reward = 1-self.total_reward_accumulated
			#reward = 0
			self.total_reward_accumulated+= 1

			return 1, done

		else:
			cur_dist = self.calculate_distance(self.state[0] , self.state[1])

			diff_dist = self.old_dist - cur_dist
			reward = diff_dist/self.total_distance
			#reward = 0
			self.total_reward_accumulated+= reward
			return reward,done


if __name__ == "__main__":
	print("ddd")
	cb = createBoard(display=True)
	for i in range(100):
		print("Here")
		cb.reset()
		for j in range(300000):
			if cb.display:
				cb.render()
			action = cb.take_action_from_user()
			print(action)
			state ,reward ,done , _ = cb.step(action)
			if done:
				break
		print(reward) 
		print(cb.total_reward_accumulated)
		
	pygame.quit()

c
