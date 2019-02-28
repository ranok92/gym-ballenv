# basic CartPole simulation

import gym , sys , time
import gym_ballenv
import numpy as np
import readchar
#from pyglet.window import key
from queue import Queue
import os
import argparse
from itertools import count
import math
import PIL
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torchvision.transforms as T
from PIL import Image
import time

from torch.autograd import Variable

#from pyvirtualdisplay import Display
from collections import namedtuple

#display =Display(visible=0,size=(1400,900))
#display.start()
#_screen_height = 100
#_screen_width = 100
#_normalizing_dist = math.sqrt(pow(_screen_width,2)+pow(_screen_height,2))

def read_arguments():

    parser = argparse.ArgumentParser(description='Insert details about the Environment :')
    parser.add_argument('--static_obstacles' , default = 13 , type = int , help='No of static obstacles to be deployed in the environment')
    parser.add_argument('--dynamic_obstacles' , default = 5 , type=int , help='No. of dynamic obstacles to be deployed in the environment')
    parser.add_argument('--obstacle_speed', default= [1,1,1,1,1], nargs='+' , help = 'List of speed for the dynamic obstacles')
    parser.add_argument('--obs_goal_position' , nargs='+', default = ['12,122','123,93','87,150','430,440','230,11'] , help="List of goal positions for the dynamic obstacles. Use this format 'x_coord,y_coord'")
    parser.add_argument('--time_step_for_change' , default = 50 , type=int)
    #parser.add_arguement('--')
    parser.add_argument('--rd_th_obs' , type=int , default=60 , help='Centainity in the action taken by the obstacles(0-100 where 100 means no uncertainty)')
    parser.add_argument('--rd_th_agent' , type=int , default=80 , help='Centainity in the action taken by the agent(0-100 where 100 means no uncertainty)')
    parser.add_argument('--static_thresholds' , nargs='+' , default = [0,0] , type=int , help='Penalty thresholds for static obstacles.')
    parser.add_argument('--dynamic_thresholds' , nargs='+' , default = [10,10] , type=int , help='Penalty thresholds for dynamic obstacles')
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

class MapCoords():
    def __init__(self,prob_arr,action):

        self.prob_arr = prob_arr
        self.action_taken = action



class PolicyCNN(nn.Module):
    def __init__(self):
        super(PolicyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        self.action_head = nn.Linear(132, 9)
        self.value_head = nn.Linear(132,1)
        self.saved_actions = []
        self.rewards = []

    def forward(self, x , y):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)
        #print(x.size())
        #print(y.size())
        x = torch.cat((x,y),1)
        a_x = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(a_x, dim= -1) , state_values


class Policy(nn.Module):
    
    def __init__(self, window):
        super(Policy, self).__init__()

        no_of_l1Nodes = 4+(window*window)
        #self.hidden_layer = no_of_l1Nodes*2
        self.hidden_layer = 128
        #self.hidden_layer =128
        self.fc1 = nn.Linear(no_of_l1Nodes, self.hidden_layer)
        #self.fc2 = nn.Linear(self.hidden_layer,self.hidden_layer2)
        #self.dropout = nn.Dropout(.5)
        self.action_head = nn.Linear(self.hidden_layer,9)
        self.value_head = nn.Linear(self.hidden_layer,1)
        #self.container_size = 5
        #self.lstm1 = nn.LSTM((self.container_size,1,128),(1,1,128))
        
        #self.state_container = Queue(maxsize=self.container_size)
        self.saved_actions = []
        self.rewards = []


    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc2(x))
        #print(x.size())
        #print(y.size())
        #x = torch.cat((x,y),1)
        #self.state_container.put(x)
        #v = torch.from_numpy(np.asarray(list(self.state_container))).to(device)
        #x = self.lstm1(v)
        a_x = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(a_x, dim= -1) , state_values

class PolicyLSTM(nn.Module):
    
    def __init__(self,timesteps):
        super(PolicyLSTM, self).__init__()

        no_of_l1Nodes = 1+28
        self.fc1 = nn.Linear(no_of_l1Nodes, 128)
        #self.fc2 = nn.Linear(128,128)
        self.dropout = nn.Dropout(.5)
        self.action_head = nn.Linear(128,9)
        self.value_head = nn.Linear(128,1)
        self.container_size = timesteps
        self.lstm1 = nn.LSTM(128,128)
        self.saved_actions = []
        self.rewards = []
        self.hidden = self.init_hidden()

    def init_hidden(self):

        return (torch.zeros(1,1,128).type(torch.FloatTensor).to(device),torch.zeros(1,1,128).type(torch.FloatTensor).to(device))

    def forward(self, state_container):

        states = []
        state_container = Variable(state_container, requires_grad =True)
        #print("state container size",state_container.size())
        #x = torch.zeros(state_container.size()[0],1,128).to(device)
        for i in range(self.container_size):
            xval = state_container[i,:]
            xj = self.dropout(F.relu(self.fc1(xval)))
            xy = xj.view(1,-1)
            states.append(xy)
        stateval = torch.stack(states,0)
        #stateval.unsqueeze(dim=1)
        #print(stateval.size())
        #x = F.relu(self.fc2(x))
        #print(x.size())
        #print(y.size())
        #x = torch.cat((x,y),1)
        #self.state_container.put(x.detach().cpu().numpy())
        #v = np.asarray(list(self.state_container.queue))
        #print(type(v))
        #print(v)
        #v = torch.from_numpy(v)
        y,self.hidden= self.lstm1(stateval,self.hidden)
        #print(y.size())
        #p = y[-1,:,:]
        a_x = self.action_head(y[-1].view(1,-1))
        state_values = self.value_head(y[-1].view(1,-1))
        return F.softmax(a_x, dim= -1) , state_values



def select_actionCNN(state,sec_part):
 
    probs, state_value = policy(state,sec_part)
    m = Categorical(probs)
    action = m.sample()
    #print(m.log_prob(action))
    policy.saved_actions.append(SavedAction(m.log_prob(action),state_value))
    return action.item()

def select_action(state):
    state = state.float()
    probs, state_value = policy(state)
    m = Categorical(probs)
    
    action = m.sample()
    #print(m.log_prob(action))
    #print(action)
    #print(action.item())
    policy.saved_actions.append(SavedAction(m.log_prob(action),state_value))
    return action.item(),probs

def finish_episode():
    R = 0
    saved_actions = policy.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards).to(device)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.item()
        policy_losses.append(-log_prob * reward)
        #print("aaaa",torch.tensor([r]).to(device).size())
        #print("sccc",value.size())

        value_losses.append(F.smooth_l1_loss(value.squeeze(), torch.tensor([r]).squeeze().to(device)))
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    print("Loss",loss)
    loss.backward(retain_graph =True)
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_actions[:]

def extract_patch(state,width=100):
    np.set_printoptions(threshold=np.nan)
    #vdisplay = Xvfb(width = 200, height = 200)
    #vdisplay.start()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resize = T.Compose([T.ToPILImage(),
                    T.Resize((40,40), interpolation=PIL.Image.BILINEAR),
                    T.ToTensor()])
    #print("State info",state)
    span = width/2

    
    screen = env.render(mode='rgb_array').copy()
    
    print("The screen",screen)
    time.sleep(2)
    img = Image.fromarray(screen.astype('uint8'),'RGB')
    img.show()
    #time.sleep(10)
    agent_x = state[0][0]+span
    #print(screen.shape[0])
    agent_y = screen.shape[0]-state[0][1]+span
    screen = nn.functional.pad(torch.from_numpy(screen),(0,0,span,span,span,span),mode='constant',value=255)

    screen = screen.numpy()
    #screen = screen.numpy()

    
    #print(screen[agent_y,agent_x,:])

    #screen = screen.permute(1,0,2)
    #print(screen.shape)
    #patch = screen[agent_y-span:agent_y+span,agent_x-span:agent_x+span,:]
    #print(patch.shape)
    #patch = np.ascontiguousarray(patch, dtype=np.float32)
    #plt.imshow(patch)
    ###print(patch.shape)
    

    patch = torch.from_numpy(screen)
    print("The screen2",screen)
    time.sleep(2)
    patch = resize(patch).unsqueeze(0).to(device)
    img = Image.fromarray(patch.squeeze().cpu().numpy().astype('uint8'),'RGB')
    img.show()
    print("First patch",patch)
    #img = Image.fromarray(patch.numpy().astype('uint8'),'RGB')
    #img.show()
    #time.sleep(10)

    ###print(patch.shape)
    #patch = patch.permute(2,0,1).numpy()
    #patch = T,Resize()
    #print(patch.shape)
    

    #plt.imshow(torch.from_numpy(patch))#.permute(1,2,0))
    #time.sleep(12)
    return patch 


def prep_state(state):
    #print("insidePrepState",state)
    val = math.sqrt(math.pow(_screen_width,2)+math.pow(_screen_height,2))
    prep_state = []
    state = list(state)
    for i in state:
        #print(type(i))
        if isinstance(i,tuple):
            #print(i)
            nlist = list(i)
            nlist[0] = float(nlist[0])/_screen_width
            nlist[1] = float(nlist[1])/_screen_height
            for j in nlist:
                prep_state.append(j)
        else:
            prep_state.append(i/val)

    return torch.from_numpy(np.asarray(prep_state)).to(device).unsqueeze(0)


def prep_state2(state):

#state is a list of info where 1st position holds the position of the 
#agent, 2nd the position of the goal , 3rd the distance after that, 
#the positions of the obstacles in the world
    #print(state)
    ref_state = np.zeros(4)
    #print(ref_state.shape)
    #ref_state[4] = 1
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

    return torch.from_numpy(ref_state).float().to(device).unsqueeze(0)

def block_to_arrpos(x,y):

    pos = 12+(5*y)+x
    return pos

def calculate_distance(tup1, tup2):

    x_diff = tup1[0] - tup2[0]
    y_diff = tup1[1] - tup2[1]
    return math.sqrt(math.pow(x_diff,2)+math.pow(y_diff,2))

#distance from each of the obstacle in the board
def prep_state3(state):
    
    ref_state = np.zeros(len(state)-2)
    agent_pos = state[0]
    ref_state[0] = state[2]/_normalizing_dist
    c = 1
    for i in range(3,len(state)):
        obs_pos = state[i]
        dist_to_collosion = calculate_distance(agent_pos,obs_pos)-env.unwrapped.radius_rand_person-env.unwrapped.radius_ctrl_person
        ref_state[c] = dist_to_collosion/_normalizing_dist
        c+=1
    return torch.from_numpy(ref_state).float().to(device).unsqueeze(0)

#creates a state representation in the form of a 25+4 vector
#the 25 vector is a 5x5 matrix centered around the agent which denotes which
#boxes are occupied by an obstacle and the 4 vector denotes the general position in 
#which the goal is present. The size of the blocks of the matrix is denoted
#by the speed of the agent
def prep_state4(state,window):

    ref_state = np.zeros(4+(window*window))
    ref_state[0:4] = prep_state2(state)
    counter = 4
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
    return torch.from_numpy(ref_state).float().to(device).unsqueeze(0)

def update_MAP(map,coordinates, action_distribution, action_taken , exploration ,k):

    if coordinates not in map:

        map[coordinates] = MapCoords(action_distribution, action_taken)
        return action_taken

    else:


        #print ("Updating cooridinates :", coordinates)
        #print ("Prob :", map[coordinates].prob_arr)
        #print ("Old action :", map[coordinates].action_taken)
        action_prob = map[coordinates].prob_arr.clone()

        #print(map[coordinates].prob_arr)
        #print(map[coordinates].prob_arr.size())
        #print(map[coordinates].action_taken)
        action_taken = map[coordinates].action_taken
        value = map[coordinates].prob_arr[0,action_taken].clone()
        #print("value",value)
        index = action_taken
        #value,index = action_prob.max(0)
        newVal = value*(1-exploration)
        #print("Action prob before",action_prob)
        action_prob[0,index] = -1
        #print("Action prob after",action_prob)
        topvals,topindex = action_prob[0].topk(k)
        #print ("TOPvals",topvals)
        #print("Topindexes", topindex)
        #print("ASSs",value)
        for i in range(k):
            topvals[i] = topvals[i] + value*exploration/k
        #print ("new topvals",topvals)
        for i in range(len(topindex)):
            action_prob[0,topindex[i]] = topvals[i]
        action_prob[0,index] = newVal
        map[coordinates].prob_arr = action_prob
        m = Categorical(action_prob)
        map[coordinates].action_taken = m.sample().item()
        #print(action_prob)
        #print (map[coordinates].prob_arr)
        #print (map[coordinates].action_taken)
        #exit()
        #print (" New Prob :", map[coordinates].prob_arr)
        #print ("New action :", map[coordinates].action_taken)
    return map[coordinates].action_taken

args = read_arguments()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device :",device)
assert_arguments(args)
#print('sssss',type(args.obs_goal_position[1]))
print(args.static_obstacles)
env = gym.make('gymball-v0') # create the environment
#print(type(env.unwrapped))

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

######MAP TO STORE ACTIONS##########
ACTIONHISTORY = {}
EXPLORATION = .4
KVAL = 3
####################################
TIMESTEPS = 5
env.unwrapped.customize_environment(args)
mode = 'policy'
env.reset()
#extract_patch()
#time.sleep(4)
SKIP_CONTROL = 0
human_agent_action = np.array([0,0])
human_wants_restart = False
human_sets_pause = False
f = None
ACTIONS = env.action_space.n

###crete the network#############
#policy = Policy(args.static_obstacles)
WINDOW = 5
policy = Policy(WINDOW)
#policy = PolicyCNN()
#policy = PolicyLSTM(TIMESTEPS)
#################################

if torch.cuda.is_available():
    policy.cuda()

if args.resume:
    checkpoint = torch.load(args.resume)
    policy.load_state_dict(checkpoint)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
eps = np.finfo(np.float32).eps.item()





xs = []
ys = []
yrm = []



style.use('fivethirtyeight')

#ax1 = fig.add_subplot(1,1,1)


def animate(i):
    ax1.clear()
    ax1.plot(xs,ys)


def main():
    torch.set_printoptions(threshold=5000)
    move_list = [(1,1) , (1,-1) , (1, 0) , (0,1) , (0,-1),(0,0) , (-1,1),(-1,0),(-1,-1)]
    running_reward = 10
    state_container = torch.zeros(TIMESTEPS,29).to(device)
    #ani = animation.FuncAnimation(fig, animate , interval=1000)
    plt.ion()
    print("Using", torch.cuda.is_available())
    for i_episode in count(1):
        ACTIONHISTORY = {}

        if i_episode>=1002:
            exit()
        state = env.reset()
        xrwd = []
        yrwd = []
        #print("AgentPos",state[0])
        #print("Goal_pos",state[1])
        env.render()
        #print("STate",state)
        print("Running episode :",i_episode)
        for b in range(1):
            #print b
            #t = 0
            state = env.reset()
            for t in range(12000):  # Don't infinite loop while learning
                #if t==1888:
                env.render()
                ######prepare state#########
                #patch = extract_patch(state)
                #sec_part = prep_state2(state)
                #nonCNN part
                state_tensor = prep_state4(state,WINDOW)
                ##state_container[0:4] = state_container[1:5]
                ##state_container[4] = state_tensor
                #print "The statetensor",state_tensor
                #exit()
                ######LSTM PART#####
                #state_container[0:TIMESTEPS-1] = state_container[1:TIMESTEPS]
                #state_container[TIMESTEPS-1] = state_tensor

                ############################

                #########take action###########
                action,probs = select_action(state_tensor)
                action = move_list[action]
                #_,action = probs.topk(1)
                #print action
                ####work with map##########
                ###actionupdated = update_MAP(ACTIONHISTORY,state[0],probs, action, EXPLORATION , KVAL)
                ###action = move_list[actionupdated]
                #action = select_actionCNN(patch,sec_part)
                #action = move_list[action]
                #######take action for LSTM###########
                #action = select_action(state_container)
                #print(action)
                #action = move_list[action]
                ################################

                ##print("ACtion",action)
                state, reward, done, _ = env.step(action)
                #print("STATE",state)
                #if args.render:
                if i_episode+1%100==0: #and (b+1)%11==0:
                    #print b
                    '''
                    fig2 = plt.figure(2)
                    plt.clf()
                    #print("The state",state_prep)
                    #display the reward function for the run
                    
                    xrwd.append(t)
                    yrwd.append(env.unwrapped.total_reward_accumulated)
                    plt.plot(xrwd,yrwd,color='black')
                    plt.draw()
                    plt.pause(.0001)
                    plt.title("Reward function for run:")
                    fig2.show()
                    '''
                    #time.sleep(.5)
                    env.render()
     
                policy.rewards.append(reward)
                if done:
                    print("Steps taken :{} Reward obtained {}".format(t,env.unwrapped.total_reward_accumulated))
                    break
        if t>0:
            finish_episode()
            ppp=1
        else:
            print("no steps taken. Nothing to train")
        #running_reward = running_reward * 0.99 + t * 0.01
        #plotting the reward for each of the 10th iteration
        if (i_episode%1==0 and t>0):
            #plotting the reward for each run
            fig =plt.figure(1)
            xs.append(i_episode)
            ys.append(env.unwrapped.total_reward_accumulated)
            yrm.append(sum(ys)/len(ys))
            plt.plot(xs,ys,color='green',linewidth = .5)
            plt.plot(xs,yrm,color='red',linestyle='dashed')
            plt.draw()
            plt.title("Reward function across multiple runs:")
            plt.pause(.0001)
            fig.show()
            #plotting the loss for each run 


        
        if i_episode % 500==0:

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
    


