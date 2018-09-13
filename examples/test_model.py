# basic CartPole simulation

import gym , sys , time
import gym_ballenv
import numpy as np
import readchar
from pyglet.window import key
import os
import argparse
from itertools import count
import math

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import time

def read_arguments():

    parser = argparse.ArgumentParser(description='Insert details about the Environment :')
    parser.add_argument('--static_obstacles' , default = 10 , type = int , help='No of static obstacles to be deployed in the environment')
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

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        #self.affine1 = nn.Linear(39, 128)
        self.affine1 = nn.Linear(29, 128) #for prep_state2
        self.affine2 = nn.Linear(128,128)
        self.affine3 = nn.Linear(128, 9)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        x = F.relu(self.affine3(x))
        #print("ACTION",action_scores)
        #action_scores = F.normalize(action_scores)
        return x


def select_action(state):
    #print("state",state)
    state = torch.from_numpy(state).float().unsqueeze(0)
    #print("State",state)
    probs = policy(state)
    #print("Probs",probs)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()
    return action


def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        #print(policy_loss)
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    #print("Policy_LOss",policy_loss)
    policy_loss = torch.cat(policy_loss).sum()
    print("Policy_loss",policy_loss)
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]
    #Storch.cuda.empty_cache()

def prep_state(state):
    #print("insidePrepState",state)
    val = math.sqrt(math.pow(640,2)+math.pow(480,2))
    prep_state = []
    state = list(state)
    for i in state:
        #print(type(i))
        if isinstance(i,tuple):
            #print(i)
            nlist = list(i)
            nlist[0] = float(nlist[0])/640
            nlist[1] = float(nlist[1])/480
            for j in nlist:
                prep_state.append(j)
        else:
            prep_state.append(i/val)

    return np.asarray(prep_state)


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

def block_to_arrpos(x,y):

    pos = 12+(5*y)+x
    return pos

args = read_arguments()

assert_arguments(args)
#print('sssss',type(args.obs_goal_position[1]))

#print(args)
env = gym.make('gymball-v0') # create the environment
#print(type(env.unwrapped))
env.unwrapped.customize_environment(args)
mode = 'policy'
env.reset()
SKIP_CONTROL = 0
human_agent_action = np.array([0,0])
human_wants_restart = False
human_sets_pause = False
f = None
ACTIONS = env.action_space.n

policy = Policy()
if args.resume:
    checkpoint = torch.load(args.resume)
    policy.load_state_dict(checkpoint)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()



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
            #time.sleep(0.2)
        #time.sleep(0.2)
    print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))
    return False

xs = []
ys = []
yrm = []



style.use('fivethirtyeight')

#ax1 = fig.add_subplot(1,1,1)


def animate(i):
    ax1.clear()
    ax1.plot(xs,ys)


def main():
    move_list = [(1,1) , (1,-1) , (1, 0) , (0,1) , (0,-1),(0,0) , (-1,1),(-1,0),(-1,-1)]
    running_reward = 10
    #ani = animation.FuncAnimation(fig, animate , interval=1000)
    plt.ion()
    print("Using", torch.cuda.is_available())
    for i_episode in count(1):
        state = env.reset()
        xrwd = []
        yrwd = []
        print("AgentPos",state[0])
        print("Goal_pos",state[1])
        #print("STate",state)
        for t in range(1000):  # Don't infinite loop while learning

            state_prep = prep_state2(state)
            #time.sleep(1)
            ##print("The state", state_prep)
            action = select_action(state_prep)
            action = move_list[action]
            ##print("ACtion",action)
            state, reward, done, _ = env.step(action)
            #print("STATE",state)
            #if args.render:
            if (i_episode%1==0):
                fig2 = plt.figure(2)
                plt.clf()
                print("The state",state_prep)
                #display the reward function for the run
                if t%10==0:
                    xrwd.append(t)
                    yrwd.append(env.unwrapped.total_reward_accumulated)
                    plt.plot(xrwd,yrwd,color='black')
                    plt.draw()
                    #plt.pause(.0001)
                    plt.title("Reward function for run:")
                    fig2.show()
                #time.sleep(.5)
                env.render()
 
            policy.rewards.append(reward)
            if done:
                break

        #running_reward = running_reward * 0.99 + t * 0.01
        #plotting the reward for each of the 10th iteration
        if (i_episode%10==0):
            #plotting the reward for each run
            fig =plt.figure(1)
            xs.append(i_episode)
            ys.append(env.unwrapped.total_reward_accumulated)
            yrm.append(sum(ys)/len(ys))
            plt.plot(xs,ys,color='green')
            plt.plot(xs,yrm,color='red',linestyle='dashed')
            plt.draw()
            plt.title("Reward function across multiple runs:")
            plt.pause(.0001)
            fig.show()
            #plotting the loss for each run 


        #finish_episode()
        if i_episode % 1000==0:

            torch.save(policy.state_dict(),'./stored_models/episode_{}.pth'.format(i_episode))
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, env.unwrapped.total_reward_accumulated))
        if running_reward > 10000:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break




if __name__ == '__main__':
    main()

#print(totals)
#print('The longest number of timesteps the pole was balanced: ' + str(max(totals)))
    


