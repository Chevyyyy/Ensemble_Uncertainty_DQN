import gym
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from deep_endemble_NN_model import GaussianMixtureMLP
from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
DQN_model=GaussianMixtureMLP(5,2,3)

DQN_model.load_state_dict(torch.load("models/ensemble_FE.pt"))


def getRM(model,plot=True,filename="RM.png"):
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
        RM=RM.max(1)[1]

        plt.scatter(points[:,0], points[:,1], c=RM,cmap="viridis")


        plt.colorbar()
        plt.savefig(f"imgs\{filename}")
        plt.show()

        if plot:
            plt.show()
        plt.close()
        
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = 0.1

    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return DQN_model(state)[0].max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], dtype=torch.long)


# getRM(DQN_model,False,"300.png")

count=0
env = gym.make("MountainCar-v0",render_mode="human")
with torch.no_grad():
    obs, _ = env.reset()
    done = False
    while not done:
        count=count+1
        # step forward
        action=select_action(torch.tensor([obs]))
        # GetParticles(_,reward_model)
        obs, reward, done, _, _ = env.step(action.item())
        print("step",count,"#",action)

