import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from scipy.stats import norm



def Thompson_sampling(means,vars):
    n=len(means)
    R=[]
    for k in range(n):
        R.append(np.random.normal(means[k],vars[k]**0.5))
    R=np.array(R)
    return torch.tensor(np.argmax(R)).unsqueeze(0)
    

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
        RM=RM[0].max(1)[1]

        plt.scatter(points[:,0], points[:,1], c=RM,cmap="viridis")


        plt.colorbar()
        plt.savefig(f"imgs\{filename}")
        plt.show()

        if plot:
            plt.show()
        plt.close()


pi = torch.tensor(np.pi)







class GaussianMultiLayerPerceptron(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout=nn.Dropout(p=0) 
        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.output_dim)
        
    def forward(self, x):
        batch_n=x.shape[0]
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x).reshape(batch_n,2,-1)
        mean=x[:,0,:]
        variance=x[:,1,:]
        variance = F.softplus(variance) +1e-6  #Positive constraint
        return mean, variance
    
class GaussianMixtureMLP(nn.Module):
    """ Gaussian mixture MLP which outputs are mean and variance.

    Attributes:
        models (int): number of models
        inputs (int): number of inputs
        outputs (int): number of outputs
        hidden_layers (list of ints): hidden layer sizes

    """
    def __init__(self, num_models=5, inputs=1, outputs=1):
        super(GaussianMixtureMLP, self).__init__()
        self.num_models = num_models
        self.inputs = inputs
        self.outputs = outputs
        for i in range(self.num_models):
            model = GaussianMultiLayerPerceptron(self.inputs,self.outputs*2)
            setattr(self, 'model_'+str(i), model)
            optim=torch.optim.AdamW(getattr(self, 'model_' + str(i)).parameters(),lr=0.0001)
            setattr(self,"optim_"+str(i),optim)
            
    def forward(self, x):
        self.eval()
        # connect layers
        means = []
        variances = []
        for i in range(self.num_models):
            model = getattr(self, 'model_' + str(i))
            mean, var = model(x)
            means.append(mean)
            variances.append(var)
        means = torch.stack(means)
        mean = means.mean(dim=0)
        variances = torch.stack(variances)
        variance = (variances + means.pow(2)).mean(dim=0) - mean.pow(2)
        variance=F.relu(variance)+1e-6
        return mean, variance
    
    def optimize(self,x_M,t_M):
        self.train()
        for i in range(self.num_models):
            model = getattr(self, 'model_' + str(i))
            optim = getattr(self, 'optim_' + str(i))
            # forward
            mean, var = model(x_M[i])
            # compute the loss
            optim.zero_grad()
            
            loss=F.gaussian_nll_loss(mean,t_M[i],var)
            # optimize
            loss.backward()
            optim.step()

    def optimize_replay(self,current_state,next_state,action,reward,non_final_mask,gamma,target_net):
        batch_n=current_state.shape[0]

    
        
        current_state=current_state.reshape(self.num_models,int(batch_n/self.num_models),-1)
        next_state=next_state.reshape(self.num_models,int(batch_n/self.num_models),-1)
        reward=reward.reshape(self.num_models,int(batch_n/self.num_models),-1)
        action=action.reshape(self.num_models,int(batch_n/self.num_models),-1)
        non_final_mask=non_final_mask.reshape(self.num_models,int(batch_n/self.num_models),-1)
        
        
        
        self.train()
        for i in range(self.num_models):
            model = getattr(self, 'model_' + str(i))
            optim = getattr(self, 'optim_' + str(i))
            target_model=getattr(target_net, 'model_' + str(i))
            # forward
            mean, var = model(current_state[i])
            # compute the loss
            optim.zero_grad()

            state_action_values = mean.gather(1, action[i]).squeeze()
            state_action_values_var = var.gather(1, action[i]).squeeze()

            with torch.no_grad():
                next_state_values = torch.zeros(int(batch_n/self.num_models))
                next_state_values[non_final_mask[i].squeeze()] = target_model(next_state[i])[0].max(1)[0]
            
            loss=F.gaussian_nll_loss(state_action_values,next_state_values+gamma*reward[i].squeeze(),state_action_values_var)
            # optimize
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 100)
            optim.step()
            
        
        
        
# sns.set_style("white")
# test_ratio = 0.0
# data_step = 0.005
# data_sigma1 = 2
# data_sigma2 = 1
# def func(x):
#     return np.power(x, 3)

# num_data = 20

# data_x = np.random.uniform(-4, 4, size=num_data)
# data_y = np.zeros(num_data)

# num_data_true = 1000
# data_x_true = np.linspace(-6, 6, num_data_true)
# data_y_true = np.zeros(num_data_true)
# for i in range(num_data):
#     if (data_x[i] < 0):  # -3 <= x <0, sigma=2 (has more uncertainty inherently)
#         data_y[i] = func(data_x[i]) + np.random.normal(0, 3)
#     else:  # x>0, sigma=1 (less noisy measurement)
#         data_y[i] = func(data_x[i]) + np.random.normal(0, 3)
        
# for i in range(num_data_true):
#     data_y_true[i] = func(data_x_true[i])

# num_train_data = int(num_data * (1 - test_ratio))
# num_test_data  = num_data - num_train_data

# data_x = np.reshape(data_x, [num_data, 1])
# data_y = np.reshape(data_y, [num_data, 1])
# data_y_true = np.reshape(data_y_true, [num_data_true, 1])
# data_x_true = np.reshape(data_x_true, [num_data_true, 1])

# train_x = data_x[:num_train_data, :]
# train_y = data_y[:num_train_data, :]
# test_x  = data_x[num_train_data:, :]
# test_y  = data_y[num_train_data:, :]
# tensor_x = torch.Tensor(train_x) # transform to torch tensor
# tensor_y = torch.Tensor(train_y)

# from torch.utils.data import TensorDataset, DataLoader
# toy_dataset = TensorDataset(tensor_x, tensor_y) # create your datset
# model_mine=GaussianMixtureMLP(num_models=5)

# dataloader = DataLoader(toy_dataset, batch_size=3, shuffle=True) # create your dataloader


# for _ in tqdm(range(1000)):
#     x_M=[]
#     t_M=[]
#     for i in range(5):
#         x,y=next(iter(dataloader))
#         x_M.append(x)
#         t_M.append(y)
#     x_M=torch.cat(x_M,1).T.unsqueeze(-1)
#     t_M=torch.cat(t_M,1).T.unsqueeze(-1)
#     model_mine.optimize(x_M,t_M)
    
# mean=[]
# variance=[]
# for x in data_x_true:
#     mu, sigma = model_mine(torch.Tensor(x).unsqueeze(0))
#     mean.append(mu)
#     variance.append(sigma)
# variance=torch.tensor(variance)
# mean=torch.tensor(mean)
# std_devs = np.sqrt(variance.numpy()) * 3

# upper = [i + k for i, k in zip(mean.numpy(), std_devs)]
# lower = [i - k for i, k in zip(mean.numpy(), std_devs)]

# plt.rcParams['figure.figsize'] = [8, 7]
# plt.axvline(x=0, linewidth=2)
# plt.plot(data_x_true, mean, '.', markersize=6, color='#F39C12')
# plt.fill_between(data_x_true.flatten(), upper, lower, color="orange", alpha=0.4)
# plt.plot(data_x_true, data_y_true, 'r', linewidth=3)
# plt.plot(data_x, data_y, '.', markersize=12, color='#F39C12')
# plt.legend(['Data', 'y=x^3'], loc = 'best')
# plt.show()