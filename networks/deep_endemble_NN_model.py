# define the class for ensemble DQN network
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaussianMultiLayerPerceptron(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout=nn.Dropout(p=0) 
        self.fc1 = nn.Linear(self.input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.output_dim)
        
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

    def optimize_replay(self,current_state,next_state,action,reward,gamma,target_net):
        batch_n=current_state.shape[0]

    
        
        current_state=current_state.reshape(self.num_models,int(batch_n/self.num_models),-1)
        reward=reward.reshape(self.num_models,int(batch_n/self.num_models),-1)
        action=action.reshape(self.num_models,int(batch_n/self.num_models),-1)
        

        
        
        self.train()
        for i in range(self.num_models):

            next_state_i=next_state[int(i*batch_n/self.num_models):int((i+1)*batch_n/self.num_models)]

            
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            next_state_i)), dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in next_state_i
                                                    if s is not None])
            
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
                next_state_values[non_final_mask.squeeze()] = target_model(non_final_next_states)[0].max(1)[0]
            
            loss=F.gaussian_nll_loss(state_action_values,next_state_values+gamma*reward[i].squeeze(),state_action_values_var)
            # optimize
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 100)
            optim.step()
            
        
        
        
