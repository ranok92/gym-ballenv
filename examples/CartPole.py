# basic CartPole simulation

import gym , sys , time
import gym_ballenv
import numpy as np
import readchar
from pyglet.window import key
import os
import argparse


def read_arguments():

    parser = argparse.ArgumentParser(description='Insert details about the Environment :')
    parser.add_argument('--static_obstacles' , default = 5 , type = int , help='No of static obstacles to be deployed in the environment')
    parser.add_argument('--dynamic_obstacles' , default = 4 , type=int , help='No. of dynamic obstacles to be deployed in the environment')
    parser.add_argument('--obstacle_speed', default= [10,10,10,10], nargs='+' , type=int , help = 'List of speed for the dynamic obstacles')
    parser.add_argument('--obs_goal_position' , nargs='+', default = [ (300,400) , (300,400) , (600,300) , (600,300) ] , help="List of goal positions for the dynamic obstacles. Use this format 'x_coord,y_coord'")
    parser.add_argument('--time_step_for_change' , default = 50 , type=int)
    #parser.add_arguement('--')
    parser.add_argument('--rd_th_obs' , type=int , default=60 , help='Centainity in the action taken by the obstacles(0-100 where 100 means no uncertainty)')
    parser.add_argument('--rd_th_agent' , type=int , default=80 , help='Centainity in the action taken by the agent(0-100 where 100 means no uncertainty)')
    parser.add_argument('--static_thresholds' , nargs='+' , default = [10000,2500] , type=int , help='Penalty thresholds for static obstacles.')
    parser.add_argument('--dynamic_thresholds' , nargs='+' , default = [10000,2500] , type=int , help='Penalty thresholds for dynamic obstacles')
    parser.add_argument('--static_penalty' , nargs='+' , default=[-10000,-50000] , type=int ,  help='Penalty suffered for crossing thresholds for static obstacles')
    parser.add_argument('--dynamic_penalty' , nargs='+' , default=[-10000,-50000] , type=int ,  help='Penalty suffered for crossing thresholds for dynamic obstacles')
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


args = read_arguments()

assert_arguments(args)
#print('sssss',type(args.obs_goal_position[1]))

#print(args)
env = gym.make('gymball-v0') # create the environment
#print(type(env.unwrapped))
env.unwrapped.customize_environment(args)
env.mode = 'human'
env.reset()
SKIP_CONTROL = 0
human_agent_action = np.array([0,0])
human_wants_restart = False
human_sets_pause = False
f = None
ACTIONS = env.action_space.n



def basic_policy(obs): # determines what action to take
    dist = obs[4]
    x = obs[0]
    goal_x = obs[2]
    y = obs[1]
    goal_y = obs[3]
    move = np.array([1,1])
    if goal_x < x:
    	move[0] = -1
    if goal_y < y:
    	move[1] = -1
    return move



def key_press(k, mod):
	print('in key press',key)
	#a = np.zeros(2)
	global human_agent_action, human_wants_restart, human_sets_pause
	if k==0xff0d: human_wants_restart = True
	if k==32: human_sets_pause = not human_sets_pause
	if k==key.UP: #65362L: #up arrow
		human_agent_action[1] = 10
	if k==key.LEFT:#65361L: #left arrow
		human_agent_action[0] = -10
	if k==key.RIGHT:#65363L:
		human_agent_action[0] = 10
	if k==key.DOWN:#65364L:
		human_agent_action[1] = -10
	#human_agent_action = a



def key_release(k, mod):
	print('in key release',k)
	global human_agent_action
	a = int( k - ord('0') )
	if k==key.UP:#65362L: #up arrow
		human_agent_action[1] = 0
	if k==key.LEFT:#65361L: #left arrow
		human_agent_action[0] = 0
	if k==key.RIGHT:#65363L:
		human_agent_action[0] = 0
	if k==key.DOWN:#65364L:
		human_agent_action[1] = 0


def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False
    obser = env.reset()
    skip = 0
    total_reward = 0
    total_timesteps = 0
    while 1:
        if not skip:
            #print("taking action {}".format(human_agent_action))
            a = human_agent_action
            total_timesteps += 1
            skip = SKIP_CONTROL
        else:
            skip -= 1

        obser, r, done, info = env.step(a)
        f.write(str(human_agent_action)+',')
        if r != 0:
            print("reward %0.3f" % r)
        total_reward += r
        window_still_open = env.render()
        if window_still_open.any()==False: return False
        if done: break
        if human_wants_restart: break
        while human_sets_pause:
            env.render()
            time.sleep(0.5)
        time.sleep(0.5)
    print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))
    return False




totals = [] # list of the total reward accumulated for each episode, 10

for episode in range(2):
    episode_rewards = 0 # the rewards for the episode, in this case just "staying alive" or running as long as possible
    print("STRTING")
    obs = env.reset() # initial obersevation, carts horizontal positon (0.0 for center), carts velocity, pole angle, angular velocity (how fast the pole is falling)  

    action = [1,1] # move the cart left or right
    if env.mode=='rgb_array':
 
    	for step in range (1000): # 1000 total steps, dont want to run forever
      
        	action = basic_policy(obs) # perform an action based on the policy and oberserved env
        	env.render()


        	obs, reward, done , info = env.step(action) # update the oberservations and reward with the action
        	print('obs',obs)
        	print('reward',reward)
        	#print('info',info)
        	episode_rewards += reward # add the reward at the current time step, 1 in this case
        	print done
        	if done:
        		totals.append(episode_rewards)
        		break
    if env.mode=='human':
    	cur_folder = os.getcwd()
    	log_list = os.listdir(os.path.join(cur_folder,'pathlogs'))
    	counter = len(log_list)+1
        print('Starting human controller version.')
        print('Controller  : Arrow keys')
        file_name = 'Trial_no_'+str(counter)
        path = os.path.join('pathlogs',file_name)
        f = open(path,'w')
        env.render()
        env.unwrapped.viewer.window.on_key_press = key_press
        env.unwrapped.viewer.window.on_key_release = key_release
        print("ACTIONS={}".format(ACTIONS))
        print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
        print("No keys pressed is taking action,", action)
        print('human_agent_action',human_agent_action)
        while 1:
        	window_still_open = rollout(env)
        	print('window_still_open',window_still_open)
        	if window_still_open==False: 
        		f.close()
        		break

#print(totals)
#print('The longest number of timesteps the pole was balanced: ' + str(max(totals)))
    


