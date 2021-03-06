import numpy as np 
import math
import gym
import gym.wrappers as wrappers
import argparse
from itertools import count
import gym_ballenv
from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot as plt 
import time


_screen_height = 100
_screen_width = 100
_normalizing_dist = math.sqrt(pow(_screen_width,2)+pow(_screen_height,2))


class continious_Environment():

	def __init__(self, width = w , height = h ,decimal_point = 100, no_static_obstacles = st_obs , rad_static_obstacles = r, no_dynamic_obstacles = dy_obs , rad_dynamic_obstacles = r_d):
		self.obstacle_list = []
		self.target = [np.random.ranf()*w , np.random.ranf()*h]
		self.agent = [np.random.ranf()*w , np.random.ranf()*h]
		self.static_obstacles = st_obs
		self.obstacle_radius = r
		self.dynamic_obstacles = dy_obs
		self.total_obstacles = st_obs + dy_obs
		self.height = h
		self.width = w
		self.deci = decimal_point
		for i in range(self.total_obstacles):
			self.obstacle_list.append([np.random.ranf()*self.width , np.random.ranf()*self.height])
		self.environment_rows = np.linspace(0 ,w , w*self.deci)
		self.environment_cols = np.linspace(0 ,h , h*self.deci)
		self.environment = np.zeros([h*self.deci , w.self.deci])

	#position is a list of length 2
	def renderObstacle(self, position , radius):
		
		position = np.asarray(position)
		position = position*self.deci
		radius = radius*self.deci
		center = [radius,radius]
		left_corner = position - center
		for r in range(2*radius):
			for c in range(2*radius):
				if np.hypot(center[0]-c,center[1]-r)<radius:
					self.environment[r+left_corner[1],c+left_corner[0]] = 10


	def renderAgent(self,position, radius):

		self.renderObstacle(position, radius)

	def renderEnvironment():

		self.environment = np.zeros([h*self.deci , w.self.deci])
		for obs in self.obstacle_list:
			self.renderObstacle(obs, self.obstacle_radius)

		self.renderAgent(self.agent,self.obstacle_radius)

		fig = plt.figure()
		ax = fig.add_subplot(111,projection='3d')
		#x = y = np.arange(0,WINDOW)
		X,Y = np.meshgrid(self.environment_cols,self.environment_rows)
		ax.plot_wireframe(X,Y,self.environment)
		plt.show()




def read_arguments():

    parser = argparse.ArgumentParser(description='Insert details about the Environment :')
    parser.add_argument('--static_obstacles' , default = 23 , type = int , help='No of static obstacles to be deployed in the environment')
    parser.add_argument('--dynamic_obstacles' , default = 0 , type=int , help='No. of dynamic obstacles to be deployed in the environment')
    parser.add_argument('--obstacle_speed', default= [], nargs='+' , help = 'List of speed for the dynamic obstacles')
    parser.add_argument('--obs_goal_position' , nargs='+', default = [] , help="List of goal positions for the dynamic obstacles. Use this format 'x_coord,y_coord'")
    parser.add_argument('--time_step_for_change' , default = 50 , type=int)
    #parser.add_arguement('--')
    parser.add_argument('--rd_th_obs' , type=int , default=60 , help='Centainity in the action taken by the obstacles(0-100 where 100 means no uncertainty)')
    parser.add_argument('--rd_th_agent' , type=int , default=80 , help='Centainity in the action taken by the agent(0-100 where 100 means no uncertainty)')
    parser.add_argument('--static_thresholds' , nargs='+' , default = [0,0] , type=int , help='Penalty thresholds for static obstacles.')
    parser.add_argument('--dynamic_thresholds' , nargs='+' , default = [10,10] , type=int , help='thresholds for dynamic obstacles')
    parser.add_argument('--static_penalty' , nargs='+' , default=[1,1] , type=int ,  help='Penalty suffered for crossing thresholds for static obstacles')
    parser.add_argument('--dynamic_penalty' , nargs='+' , default=[4000,8000] , type=int ,  help='Penalty suffered for crossing thresholds for dynamic obstacles')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
    parser.add_argument('--resume', default=False)

    args = parser.parse_args()
    return args


def assert_arguments(args):

    assert len(args.obstacle_speed)== args.dynamic_obstacles, "The length of the list of obstacle_speed does not match the no. of dynamic obstacles"
    assert len(args.obs_goal_position)==args.dynamic_obstacles , "The length of the list of obstacle_goal_position does not match the no. of dynamic obstacles"
    assert len(args.static_thresholds)==2 , "The length of the list of static_thresholds is not equal to 2"
    assert len(args.dynamic_thresholds)==2 , "The length of the list of dynamic_thresholds is not equal to 2"     #except Exception as error:
    assert len(args.static_penalty)==2 , "The length of the list of static_penalty is not equal to 2"    
    assert len(args.dynamic_penalty)==2 , "The length of the list of dynamic_penalty is not equal to 2"
    #   print error
    #    exit()


def calculate_distance(tup1, tup2):

    x_diff = tup1[0] - tup2[0]
    y_diff = tup1[1] - tup2[1]
    return math.sqrt(math.pow(x_diff,2)+math.pow(y_diff,2))



def generate_obstacle_list_from_state(state):

	obstacle_list = 10
	return obstacle_list


def prep_state4(state,window):

    ref_state = np.zeros(window*window)
    counter = 0
    agent_pos = state[0]
    step_x = env.unwrapped.speedx_ctrl_person
    step_y = env.unwrapped.speedy_ctrl_person
    start_x = agent_pos[0] - env.unwrapped.speedx_ctrl_person*int(window/2)
    start_y = agent_pos[1] - env.unwrapped.speedy_ctrl_person*int(window/2)
    cur_x = start_x
    cur_y = start_y
    for r in range(window):
        for c in range(window):
            cur_x = start_x + step_x*c
            for i in range(3,len(state)):
                #print(cur_x,cur_y)
                #print(state[i])
                #print("distance", calculate_distance((cur_x,cur_y),state[i]))
                #print('counter',counter)
                if env.unwrapped.check_overlap((cur_x,cur_y),state[i]):
                    #print("hit")
                    ref_state[counter] = 1
                    break
            counter+=1
        cur_y = start_y + step_y*r
    #print("state",state)
    #print(ref_state)
    return ref_state


def obstacle_list_from_ref_state(ref_state):

	window_size = int(math.sqrt(len(ref_state)))
	obstacle_list = []
	count = 0
	for r in range(window_size):
		for c in range(window_size):

			if ref_state[count]==1:

				x = int(count%window_size)
				y = int(count/window_size)

				obstacle_list.append((x,y))
			count+=1
	return obstacle_list

class PotentialField():
	def __init__(self,k = 5,eta = 300 ,window_size = 5 ,limit = 5 ,speed = 2):

		self.KP = k
		self.ETA = eta
		self.limit = limit
		self.window_size = window_size
		self.epsilon = 0.5
		self.pfield = None
		self.speed  = speed
		self.motion = self.get_motion_model()
		self.globalpath = {}
		self.same_state_penalty = 10
		self.normalize_force = True
		#self.obstacleList = obstacleList #obstacleList is a list of tuples 

	def calculate_positive_Potential_btwpoints(self,agent , loc_pos ,goal):

		#the value in agent is local coordinates
		#it needs to be converted to global
		#print "single run"
		#print "Global PATH :",self.globalpath
		global_coord = (agent[0]-self.window_size/2+loc_pos[0], agent[1]-self.window_size/2+loc_pos[1])
		rep_penalty = self.repeat_penalty(global_coord)
		if global_coord in self.globalpath:
			rep = self.globalpath[global_coord]
		else:
			rep = 0
		#print "Global coord : {}, Repetation : {}, Penalty : {}".format(global_coord,rep,rep_penalty)
		return 0.5*self.KP*np.hypot(global_coord[0]-goal[0],global_coord[1]-goal[1])+rep_penalty

	def calculate_negative_Potential_btwpoints(self,agent,obs):

		rho = np.hypot(agent[0]-obs[0], agent[1]-obs[1])+self.epsilon

		if rho <= self.limit:
			return 0.5*self.ETA*(1.0/rho - 1/self.limit)**2
		else:
			return 0


	def repeat_penalty(self,position):

		if position in self.globalpath:

			return self.same_state_penalty*self.globalpath[position]

		else:
			return 0

	def track_global_motion(self,agent):

		if agent not in self.globalpath:

			self.globalpath[agent] = 1
		else:

			self.globalpath[agent]+=1

	def get_motion_model(self):

		move_list = [(1,1) , (1,-1) , (1, 0) , (0,1) , (0,-1),(0,0) , (-1,1),(-1,0),(-1,-1)]
	
		return move_list


	def calculate_negative_Potential_shortest_dist(self,agent,obstacleList):

		min_dist = float("inf")
		min_point = (-1,-1)
		for i in range(len(obstacleList)):
			dist = calculate_distance(agent,obstacleList[i])
			if min_dist > dist:

				min_dist = dist
				min_point = obstacleList[i]


		return self.calculate_negative_Potential_btwpoints(agent,min_point)


	#obstacleLIst stores the relative position of the obstacles from the 
	#agent within a given windowframe

	def generate_local_Potential_Field(self, agent ,goal , obstacleList):

		self.track_global_motion(agent)
		pfield = np.zeros((self.window_size,self.window_size))
		v = self.window_size
		for r in range(v):

			for c in range(v):
				#print "Calculating potential field for {}".format(agent)

				#the local cooridinate gets converted to global coordinates
				loc_pos = (c,r)
				pfield[r,c] = self.calculate_positive_Potential_btwpoints(agent,loc_pos,goal) + self.calculate_negative_Potential_shortest_dist(loc_pos,obstacleList)

				#print (r,c)
				#print(pfield[r,c])
				#print (self.calculate_positive_Potential_btwpoints(agent,loc_pos,goal))
				#print (self.calculate_negative_Potential_shortest_dist(loc_pos,obstacleList))
		
		self.pfield = pfield
		#return pfield


	def take_step(self):

		#print "new step"
		cur_x = int(self.window_size/2)
		cur_y = int(self.window_size/2)
		#print self.pfield
		#print "The current potential", cur_potential
		candidate_action = (0,0)
		#for c in count(1):
		cur_potential = self.pfield[cur_x,cur_y]
		#print cur_potential
		for i in range(len(self.motion)):
			inx = cur_x+self.motion[i][0]*self.speed
			iny = cur_y+self.motion[i][1]*self.speed
			new_potential = self.pfield[iny,inx]
			#print "Position in the grid :x:{},y:{}".format(inx,iny)
			#print "new potential :", new_potential
			#print "Corresponding action :",self.motion[i]
			if new_potential < cur_potential:
				#print "Here"
				cur_potential = new_potential
				candidate_action = self.motion[i]

		return candidate_action


class PotentialFieldForce():
	def __init__(self,k = 5, v = 1, eta = .5 ,window_size = 50 ,limit = 20 ,speed = 2 ,weight = 5):

		self.KP = k
		self.KV = v
		self.ETA = eta
		self.limit = limit
		self.window_size = window_size
		self.epsilon = 0.5
		self.force = None
		self.agent_velocity = np.array([0,0]) #tuple (v_x , v_y)
		self.agent_weight = weight
		self.speed  = speed
		self.motion = self.get_motion_model()
		self.globalpath = {}
		self.same_state_penalty = 10
		self.normalize_force = True
		#self.obstacleList = obstacleList #obstacleList is a list of tuples 

	#Returns a vector of size 2
	def calculate_attractive_force_btwpoints(self,agent,goal):

		#the value in agent is local coordinates
		#it needs to be converted to global
		#print "single run"
		#print "Global PATH :",self.globalpath
		#global_coord = np.array([agent[0]-self.window_size/2+loc_pos[0], agent[1]-self.window_size/2+loc_pos[1]])
		global_coord = np.array(agent)
		#rep_penalty = self.repeat_penalty(global_coord)
		goal_vector = np.array(goal)

		#print "Global coord : {}, Repetation : {}, Penalty : {}".format(global_coord,rep,rep_penalty)
		attr_force =  self.KP*(global_coord - goal_vector) - self.KV*self.agent_velocity
		if self.normalize_force:

			mag = np.hypot(attr_force[0] , attr_force[1])/5
			return  - attr_force/mag
		else:
			return  - attr_force

	#this thing will return a 2D unit vector ( force along x, force along y)
	#returns a tuple
	def calculate_repulsive_force_btwpoints(self,agent,obs):

		agent = np.array([self.window_size/2 , self.window_size/2])
		rho = np.hypot(agent[0]-obs[0], agent[1]-obs[1])+self.epsilon
		print "agent :",agent
		print "obs :",obs
		print "rho :",rho
		if rho <= self.limit:
			print "here"
			force_vector_x = self.ETA*(1.0/rho - 1/self.limit)*(1/rho)*(agent[0]-obs[0])
			force_vector_y = self.ETA*(1.0/rho - 1/self.limit)*(1/rho)*(agent[1] -obs[1])
			force_mag = np.hypot(force_vector_x , force_vector_y)*4

			if self.normalize_force:
				return (force_vector_x/force_mag , force_vector_y/force_mag)
			else:
				return (force_vector_x , force_vector_y)
		else:
			return (0,0)



	def get_motion_model(self):

		move_list = [(1,1) , (1,-1) , (1, 0) , (0,1) , (0,-1),(0,0) , (-1,1),(-1,0),(-1,-1)]
	
		return move_list


	def calculate_repulsive_forces_avg(self,agent,obstacleList):

		min_dist = float("inf")
		min_point = (-1,-1)
		force_at_point = [0,0]
		for i in range(len(obstacleList)):
			#dist = calculate_distance(agent,obstacleList[i])
			force_tup = self.calculate_repulsive_force_btwpoints(agent,obstacleList[i])
			force_at_point[0] = force_at_point[0]+force_tup[0]
			force_at_point[1] = force_at_point[1]+force_tup[1]

		#print force_at_point
		#force_mag = np.hypot(force_at_point[0], force_at_point[1])
		'''
		if self.normalize_force:
			if force_mag>0:
				return np.array([force_at_point[0]/force_mag , force_at_point[1]/force_mag])
			else:
				return np.array([force_at_point[0] , force_at_point[1]])
		else:
			'''
		return np.array([force_at_point[0] , force_at_point[1]])


	#obstacleLIst stores the relative position of the obstacles from the 
	#agent within a given windowframe

	def calculate_new_velocity(self, agent ,goal , obstacleList):

		#self.track_global_motion(agent)
		
		attractive_force = self.calculate_attractive_force_btwpoints(agent,goal)
		print "Attractive_force :",attractive_force
		repulsive_force = self.calculate_repulsive_forces_avg(agent,obstacleList)
		print "Repulsive_force :",repulsive_force
		total_force = attractive_force+repulsive_force

		#print total_force
		#print self.agent_weight
		accln = total_force/self.agent_weight
		self.agent_velocity = self.agent_velocity+accln
		self.agent_velocity = total_force
		#return pfield


	def take_step(self):

		#print "new step"
		new_action = np.array([int(self.agent_velocity[0]) , int(self.agent_velocity[1])])
		return new_action



'''
pf = PotentialField()
obstacle_list = [(1,1),(4,1),(8,9),(2,2)]
goal = (5,11)
agent = (5,5)
field = pf.generate_local_Potential_Field(agent,goal,obstacle_list)
print field.shape
'''





SPEED = 1
WINDOW = 50
args = read_arguments()
env = gym.make('gymball-v0') # create the environment

#env = wrappers.Monitor(envs, './videos/' + 'gymball-v0' + '/')
env.unwrapped.customize_environment(args)

env.reset()


for i_episode in range(1):
	state = env.reset()
	print "Starting new episode ... "
	pf = PotentialFieldForce(window_size =WINDOW)

	for t in range(1000):
		ref_state = prep_state4(state,WINDOW)
		#print "The Ref_state :",ref_state
		obstacle_list = obstacle_list_from_ref_state(ref_state)
		#print "The obstacle list :",obstacle_list
		agent =  state[0]
		pf.calculate_new_velocity(agent,state[1],obstacle_list)
		#print "State",state[0]
		#print pf.globalpath
		#print pf.pfield.shape
		'''
		fig = plt.figure()
		ax = fig.add_subplot(111,projection='3d')
		x = y = np.arange(0,WINDOW)
		X,Y = np.meshgrid(x,y)
		ax.plot_wireframe(X,Y,pf.pfield)
		plt.show()
		'''
		action = pf.take_step()
		print "Action :",action

		print "action taken",action
		raw_input("keypress to continue")
		state, reward, done, _ = env.step(action)
		if i_episode%1==0:

			env.render()

		if done:
			break


