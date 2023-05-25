import torch
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import os
from buffer import Transition
import math
import random


def Thompson_sampling(means,vars):
    n=len(means)
    R=[]
    for k in range(n):
        R.append(np.random.normal(means[k],vars[k]**0.5))
    R=np.array(R)
    return torch.tensor(np.argmax(R)).unsqueeze(0)
    

def getRM(model,plot=True,filename="RM.png",ensemble=False):
    # Define the range of x and y values
    x_values = np.linspace(-1.2, 0.6, num=100)
    y_values = np.linspace(-0.07, 0.07, num=100)

    # Create a meshgrid of x and y values
    xx, yy = np.meshgrid(x_values, y_values)

    # Stack the x and y values to create a 2D array of all the 2D points
    points = np.column_stack((xx.ravel(), yy.ravel()))

    with torch.no_grad():
        model.eval()
        RM=model.forward(torch.from_numpy(points).type(torch.float32))
        if ensemble:
            RM=RM[0].max(1)[1]
        else:
            RM=RM.max(1)[1]
            

        plt.scatter(points[:,0], points[:,1], c=RM,cmap="viridis")


        plt.colorbar()
        plt.savefig(f"imgs\{filename}")

        if plot:
            plt.show()
        plt.close()

def select_action_FE(policy_net,state,PRINT_VAR_R=False):
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
    nsoftFE=np.exp(-FE)/(np.exp(-FE).sum())
    action=torch.tensor(np.argmax(np.random.multinomial(1,nsoftFE))).reshape(1,1)
    # action=torch.tensor(np.argmin(FE)).unsqueeze(0)
    
    E=0
    if action.item()-action1.item()!=0:
        E=1
        

    if PRINT_VAR_R:
        os.system("cls")
        # print("R and uncertainty:")
        print(R)
        print(var.detach().tolist())
        print(FE)
        
            
    return action,E
def select_action_FE_sample_var(policy_net,state,PRINT_VAR_R=False):
    R,var=policy_net(state)
    R=R.squeeze()
    var=var.squeeze()
    R=R.detach().tolist()
    p=[]

    for i in range(len(var)):
        try:
            p.append(np.random.normal(0,var[i].item()**0.5))
        except:
            p.append(0)
            print("var:",var[i])
    FE=-np.array(R)
    action1=torch.tensor(np.argmin(FE)).unsqueeze(0)

    FE=-np.array(R)+np.array(p)
    # nsoftFE=np.exp(-FE)/(np.exp(-FE).sum())
    # action=torch.tensor(np.argmax(np.random.multinomial(1,nsoftFE))).reshape(1,1)
    action=torch.tensor(np.argmin(FE)).unsqueeze(0)
    
    E=0
    if action.item()-action1.item()!=0:
        E=1
        

    if PRINT_VAR_R:
        os.system("cls")
        # print("R and uncertainty:")
        print(R)
        print(var.detach().tolist())
        print(FE)
        
            
    return action,E



def optimize_model_ensemble(buffer,policy_net,target_net,GAMMA=0.99,BATCH_SIZE=300,device="cpu"):
    if len(buffer) < BATCH_SIZE:
        return
    transitions = buffer.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)

    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    policy_net.optimize_replay(state_batch,batch.next_state,action_batch,reward_batch,GAMMA,target_net)


def soft_update_model_weights(policy_net,target_net,TAU):
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)


def select_action(policy_net,state,env,steps_done,device="cpu"):
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1),eps_threshold
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long),eps_threshold


def optimize_model(buffer,policy_net,optimizer,target_net,GAMMA=0.99,BATCH_SIZE=300,device="cpu"):
    if len(buffer) < BATCH_SIZE:
        return
    transitions = buffer.sample(BATCH_SIZE)
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

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()