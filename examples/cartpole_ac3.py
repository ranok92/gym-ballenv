# basic CartPole simulation

import gym , sys , time
import gym_ballenv
import numpy as np
import readchar
#from pyglet.window import key
import os
import argparse
from itertools import count
import math

import matplotlib
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

#from pyvirtualdisplay import Display
from collections import namedtuple

#display =Display(visible=0,size=(1400,900))
#display.start()
_screen_height = 100
_screen_width = 100


is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
def read_arguments():

    parser = argparse.ArgumentParser(description='Insert details about the Environment :')
    parser.add_argument('--static_obstacles' , default = 1 , type = int , help='No of static obstacles to be deployed in the environment')
    parser.add_argument('--dynamic_obstacles' , default = 0 , type=int , help='No. of dynamic obstacles to be deployed in the environment')
    parser.add_argument('--obstacle_speed', default= [], nargs='+' , help = 'List of speed for the dynamic obstacles')
    parser.add_argument('--obs_goal_position' , nargs='+', default = [] , help="List of goal positions for the dynamic obstacles. Use this format 'x_coord,y_coord'")
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


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
class PolicyCNN(nn.Module):
    def __init__(self, vgg_name):
        super(PolicyCNN, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        #self.classifier = nn.Linear(512, 10)
        self.action_head = nn.Linear(512, 2)
        self.value_head = nn.Linear(512,1)
        self.saved_actions = []
        self.rewards = []


    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                print("now")
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        #print(x.size())
        out = self.features(x)
        #print(out.size())
        out = out.view(out.size(0), -1)
        #print(x.size())
        #print(y.size())
        #x = torch.cat((x,y),1)
        #print(out.size())
        a_x = self.action_head(out)
        state_values = self.value_head(out)
        return F.softmax(a_x, dim= -1) , state_values


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128,128)
        self.action_head = nn.Linear(128,2)
        self.value_head = nn.Linear(128,1)
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #print(x.size())
        #print(y.size())
        #x = torch.cat((x,y),1)
        a_x = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(a_x, dim= -1) , state_values



def select_actionCNN(state):
 
    probs, state_value = policy(state)
    m = Categorical(probs)
    action = m.sample()
    #print(m.log_prob(action))
    policy.saved_actions.append(SavedAction(m.log_prob(action),state_value))
    return action.item()

def select_action(state):
    state = state.float().to(device)
    probs, state_value = policy(state)
    m = Categorical(probs)
    action = m.sample()
    #print(m.log_prob(action))
    policy.saved_actions.append(SavedAction(m.log_prob(action),state_value))
    return action.item()

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
    loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_actions[:]


def get_cart_location():
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


def get_screen():
    screen = env.render(mode='rgb_array').transpose(
        (2, 0, 1))  # transpose into torch order (CHW)
    # Strip off the top and bottom of the screen
    screen = screen[:, 160:320]
    view_width = 320
    cart_location = get_cart_location()
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescare, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)



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

#5x5 checker box around the agent
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

episode_durations = []
def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())



args = read_arguments()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device :",device)
assert_arguments(args)
#print('sssss',type(args.obs_goal_position[1]))

print(args.static_obstacles)
env = gym.make('CartPole-v0').unwrapped # create the environment
#print(type(env.unwrapped))

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

resize = T.Compose([T.ToPILImage(),
                    T.Resize((32,32), interpolation=Image.CUBIC),
                    T.ToTensor()])
screen_width = 600

###env.unwrapped.customize_environment(args)
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

#policy = Policy(args.static_obstacles)
policy = PolicyCNN('VGG16')
policy.cuda()
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

    move_list = [(1,1) , (1,-1) , (1, 0) , (0,1) , (0,-1),(0,0) , (-1,1),(-1,0),(-1,-1)]
    running_reward = 10
    #ani = animation.FuncAnimation(fig, animate , interval=1000)
    plt.ion()
    print("Using", torch.cuda.is_available())
    for i_episode in count(1):
        state = env.reset()
        xrwd = []
        yrwd = []
        #print("AgentPos",state[0])
        #print("Goal_pos",state[1])
        #env.render()
        #print("STate",state)
        print("Running episode :",i_episode)
        for b in range(1):
            #print b
            #t = 0
            state = env.reset()
            for t in range(1200):  # Don't infinite loop while learning
                patch = get_screen()
                #print("Patch",patch.size())
                action = select_actionCNN(patch)
                #print("Action",action)
                ##print("ACtion",action)
                #state = torch.from_numpy(state)
                #action = select_action(state)
                state, reward, done, _ = env.step(action)

                
                #print("STATE",state)
                #if args.render:
                if i_episode%20==0:
                    #print("The state",state_prep)
                    #display the reward function for the run
                    
                    #time.sleep(.5)
                    env.render()
     
                policy.rewards.append(reward)
                if done:
                    episode_durations.append(t + 1)
                    plot_durations()
                    break
        finish_episode()
        running_reward = running_reward * 0.99 + t * 0.01
        #plotting the reward for each of the 10th iteration



    

if __name__ == '__main__':
    main()

#print(totals)
#print('The longest number of timesteps the pole was balanced: ' + str(max(totals)))
    


