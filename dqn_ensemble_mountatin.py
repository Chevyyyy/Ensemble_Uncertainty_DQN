import gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from deep_endemble_NN_model import GaussianMixtureMLP,getRM,Thompson_sampling
from scipy.stats import norm
import os
import logging

logging.basicConfig(filename="log/ensemble_mountain_car_trainning.txt",
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)


PRINT_VAR_R=False
env = gym.make("MountainCar-v0")
# env = gym.make("MountainCar-v0",render_mode="human")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the AdamW optimizer
BATCH_SIZE =300
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.01

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = GaussianMixtureMLP(5,n_observations, n_actions).to(device)
target_net = GaussianMixtureMLP(5,n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

memory = ReplayMemory(10000)


steps_done = 0

def select_action_epsilon(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state)[0].max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def select_action(state):
    R,var=policy_net(state)
    R=R.squeeze()
    var=var.squeeze()
    R=R.detach().tolist()
    p=[]


    for i in range(len(var)):
        try:
            p.append(np.log(norm.pdf(0,0,var[i].item()**0.5)))
        except:
            p.append(0)
            print("var:",var[i])
    FE=-np.array(R)
    action1=torch.tensor(np.argmin(FE)).unsqueeze(0)
    FE=-np.array(R)+np.array(p)
    action=torch.tensor(np.argmin(FE)).unsqueeze(0)
    E=0
    if action.item()-action1.item()!=0:
        E=1
        

    if PRINT_VAR_R:
        os.system("cls")
        # print("R and uncertainty:")
        print(f"{R[0]:.2f}|{R[1]:.2f}|{R[2]:.2f}")
        print(f"{var[0]:.4f}|{var[1]:.4f}|{var[2]:.4f}")
        print(f"{FE[0]:.4f}|{FE[1]:.4f}|{FE[2]:.4f}")
        
        if(action.item()==0):
            print("<-")
        elif(action.item()==1):
            print("-")
        else:
            print("->")
            
    return action,E,p



episode_durations = []

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    policy_net.optimize_replay(state_batch,non_final_next_states,action_batch,reward_batch,non_final_mask,GAMMA,target_net)

num_episodes = 300

cum_R=[]
for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    E_count=0
    for t in count():
        action,E,var = select_action(state)
        E_count+=E
        observation, reward, terminated, truncated, _ = env.step(action.item())
        
        reward = torch.tensor([reward], device=device)
        done = terminated 

        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            print(i_episode," step: ",t+1)
            logging.info(f" {i_episode}, step: {t+1},E: {E_count/(t+1)}")
            print("var:",var)
            cum_R.append(t+1)
            getRM(policy_net,False,"qTable_ensemble.png")
            break

print('Complete')
plt.plot(cum_R)
plt.show()
plt.savefig("imgs/cum_R_enseble.png")
torch.save(policy_net.state_dict(),"models/ensemble_FE.pt")