# basic CartPole simulation

import gym , sys , time
import gym_ballenv
import numpy as np
import readchar
from pyglet.window import key

env = gym.make('gymball-v0') # create the environment
env.mode = 'human'
env.reset()
SKIP_CONTROL = 0
human_agent_action = np.array([0,0])
human_wants_restart = False
human_sets_pause = False

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



def key_press(key, mod):
	print('in key press',key)
	#a = np.zeros(2)
	global human_agent_action, human_wants_restart, human_sets_pause
	if key==0xff0d: human_wants_restart = True
	if key==32: human_sets_pause = not human_sets_pause
	if key==65362L: #up arrow
		human_agent_action[1] = 5
	if key==65361L: #left arrow
		human_agent_action[0] = -5
	if key==65363L:
		human_agent_action[0] = 5
	if key==65364L:
		human_agent_action[1] = -5
	#human_agent_action = a



def key_release(key, mod):
	print('in key release',key)
	global human_agent_action
	a = int( key - ord('0') )
	if key==65362L: #up arrow
		human_agent_action[1] = 0
	if key==65361L: #left arrow
		human_agent_action[0] = 0
	if key==65363L:
		human_agent_action[0] = 0
	if key==65364L:
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
        if r != 0:
            print("reward %0.3f" % r)
        total_reward += r
        window_still_open = env.render()
        if window_still_open.any()==False: return False
        if done: break
        if human_wants_restart: break
        while human_sets_pause:
            env.render()
            time.sleep(0.1)
        time.sleep(0.1)
    print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))



totals = [] # list of the total reward accumulated for each episode, 10

for episode in range(10):
    episode_rewards = 0 # the rewards for the episode, in this case just "staying alive" or running as long as possible
    obs = env.reset() # initial obersevation, carts horizontal positon (0.0 for center), carts velocity, pole angle, angular velocity (how fast the pole is falling)  

    action = [1,1] # move the cart left or right
    if env.mode!='human':
 
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
    else:
        
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
			if window_still_open==False: break

print(totals)
print('The longest number of timesteps the pole was balanced: ' + str(max(totals)))
    


