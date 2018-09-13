# basic CartPole simulation

import gym , sys , time
import gym_ballenv
import numpy as np
import readchar
from pyglet.window import key
import os
import argparse
import pickle

def read_arguments():

    parser = argparse.ArgumentParser(description='Insert details about the Environment :')
    parser.add_argument('--static_obstacles' , default = 12 , type = int , help='No of static obstacles to be deployed in the environment')
    parser.add_argument('--dynamic_obstacles' , default = 0 , type=int , help='No. of dynamic obstacles to be deployed in the environment')
    parser.add_argument('--obstacle_speed', default= [], nargs='+' , help = 'List of speed for the dynamic obstacles')
    parser.add_argument('--obs_goal_position' , nargs='+', default = [] , help="List of goal positions for the dynamic obstacles. Use this format 'x_coord,y_coord'")
    parser.add_argument('--time_step_for_change' , default = 50 , type=int)
    #parser.add_arguement('--')
    parser.add_argument('--rd_th_obs' , type=int , default=60 , help='Centainity in the action taken by the obstacles(0-100 where 100 means no uncertainty)')
    parser.add_argument('--rd_th_agent' , type=int , default=80 , help='Centainity in the action taken by the agent(0-100 where 100 means no uncertainty)')
    parser.add_argument('--static_thresholds' , nargs='+' , default = [30,20] , type=int , help='Penalty thresholds for static obstacles.')
    parser.add_argument('--dynamic_thresholds' , nargs='+' , default = [20,10] , type=int , help='Penalty thresholds for dynamic obstacles')
    parser.add_argument('--static_penalty' , nargs='+' , default=[150,300] , type=int ,  help='Penalty suffered for crossing thresholds for static obstacles')
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

def block_to_arrpos(x,y):

    pos = 12+(5*y)+x
    return pos


def prep_state2(state):

#state is a list of info where 1st position holds the position of the 
#agent, 2nd the position of the goal , 3rd the distance after that, 
#the positions of the obstacles in the world
    #print(state)
    ref_state = np.zeros(29)
    #print(ref_state.shape)
    ref_state[12+4] = 1
    agent_pos = state[0]
    goal_pos = state[1]
    diff_x = goal_pos[0] - agent_pos[0]
    diff_y = goal_pos[1] - agent_pos[1]
    if diff_x >= 0 and diff_y >= 0:
        ref_state[1] = 1
    elif diff_x < 0  and diff_y >= 0:
        ref_state[0] = 1
    elif diff_x < 0 and diff_y < 0:
        ref_state[3] = 1
    else:
        ref_state[2] = 1

    for i in range(3,len(state)):
        x_dist = agent_pos[0] - state[i][0]
        y_dist = agent_pos[1] - state[i][1]
        x_block = y_block = 0
        #print("XY",x_dist,y_dist)
        if x_dist!=0 and y_dist!=0:
            x_block = (x_dist/abs(x_dist))*(x_dist-10)//20
            y_block = (y_dist/abs(y_dist))*(y_dist-10)//20
            #print("BLOCK",x_block,y_block)
        if (abs(x_block)<3 and abs(y_block)<3): #considering a matrix of 5x5 with the agent being present at the center
        #print(x_block,y_block)
            pos = block_to_arrpos(x_block, y_block) #pos starts from 0
            #print("POS",pos)
            ref_state[4+pos] += 1

    return ref_state


args = read_arguments()

assert_arguments(args)
#print('sssss',type(args.obs_goal_position[1]))

#print(args)
env = gym.make('gymball-v0') # create the environment
#print(type(env.unwrapped))
env.unwrapped.customize_environment(args)
mode = 'human'
env.reset()
SKIP_CONTROL = 0
human_agent_action = np.array([0,0])
human_wants_restart = False
human_sets_pause = False
f = None
ACTIONS = env.action_space.n



def basic_policy(obs): # determines what action to take
    dist = obs[4]
    x = obs[0][0]
    goal_x = obs[1][0]
    y = obs[0][1]
    goal_y = obs[1][1]
    move = np.array([1,1])
    if goal_x < x:
    	move[0] = -1
    if goal_y < y:
    	move[1] = -1
    return move



def key_press(k, mod):
	print('in key press',k)
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
            #print(type(a))
            action_list.append(np.copy(a))
            #print("ACCCC",action_list)
            total_timesteps += 1
            skip = SKIP_CONTROL
        else:
            skip -= 1

        obser, r, done, info = env.step(a)
        obser = prep_state2(obser)
        print("OBser",obser)
        state_list.append(obser)
        #f2.write(str(obser)+'\n')
        #f.write(str(human_agent_action)+'\n')

        if r != 0:
            print("reward %0.3f" % r)
        total_reward += r
        window_still_open = env.render()
        if window_still_open==False: return False
        if done: break
        if human_wants_restart: break
        while human_sets_pause:
            env.render()
            time.sleep(0.2)
        time.sleep(0.2)
    print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))
    return False



totals = [] # list of the total reward accumulated for each episode, 10
state_list = []
action_list = []
for episode in range(150):
    episode_rewards = 0 # the rewards for the episode, in this case just "staying alive" or running as long as possible
    print("STRTING")
    obs = env.reset() # initial obersevation, carts horizontal positon (0.0 for center), carts velocity, pole angle, angular velocity (how fast the pole is falling)  

    #action = [1,1] # move the cart left or right
    if mode=='policy':
 
    	for step in range (1000): # 1000 total steps, dont want to run forever
      
        	action = basic_policy(obs) # perform an action based on the policy and oberserved env
        	print('Action',action)
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
    if mode=='human':
    	cur_folder = os.getcwd()
    	log_list = os.listdir(os.path.join(cur_folder,'pathlogs'))
    	counter = len(log_list)/2+1
        print('Starting human controller version.')
        print('Controller  : Arrow keys')
        file_name = 'Trial_no_'+str(counter)
        state_file_name = 'State_info_trail_no'+str(counter)
        path = os.path.join('pathlogs',file_name)
        state_path = os.path.join('pathlogs',state_file_name)
        #f = open(path,'w')
        #f2 = open(state_path,'w')
        env.render()
        env.unwrapped.viewer.window.on_key_press = key_press
        env.unwrapped.viewer.window.on_key_release = key_release
        print("ACTIONS={}".format(ACTIONS))
        print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
        #print("No keys pressed is taking action,", action)
        print('human_agent_action',human_agent_action)
        while 1:
            window_still_open = rollout(env)
            print('window_still_open',window_still_open)
            if window_still_open==False:
                #f.close()
                #f2.close()
                break
print("ACIONLIST before dump",action_list)
with open(state_file_name,'wb') as fp:
    pickle.dump(state_list,fp)
with open(file_name,'wb') as f2p:
    pickle.dump(action_list,f2p)

#print(totals)
#print('The longest number of timesteps the pole was balanced: ' + str(max(totals)))
    


