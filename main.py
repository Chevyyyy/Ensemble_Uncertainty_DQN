import gym
import matplotlib.pyplot as plt
from itertools import count
import torch
from networks.deep_endemble_NN_model import GaussianMixtureMLP
from buffer import ReplayMemory
from utilis import * 
from logger import set_log_file
import logging
from networks.DQN_model import DQN
import argparse

parser = argparse.ArgumentParser(description='|model|env|')
parser.add_argument("--model",default="ensemble_DQN",help="DQN or ensemble_DQN")
parser.add_argument("--env",default="MountainCar-v0",help="CartPole-v1|MountainCar-v0|LunarLander-v2")
parser.add_argument("--BATCH_SIZE",type=int,default=300)
parser.add_argument("--NUM_episodes",default=300)
parser.add_argument("--GAMMA",default=0.99)
parser.add_argument("--TAU",default=0.005)
parser.add_argument("--PRINT",default=False)
parser.add_argument("--render_mode",default="rgb_array")
parser.add_argument("--device",default="cpu")
parser.add_argument("--NUM_ensemble",default=5)
parser.add_argument("--file_identify",default="")
parser.add_argument("--foot_record",default=False)
args = parser.parse_args()

###############################################################################################
# config the args
# set the log file
set_log_file(f"log/{args.model}_{args.env}_{args.file_identify}.txt")
# set env
env = gym.make(args.env,render_mode=args.render_mode)
# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)
# set device
device = torch.device(args.device)

# set the model
if args.model=="DQN":
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=1e-4, amsgrad=True)
elif args.model=="ensemble_DQN":
    policy_net = GaussianMixtureMLP(args.NUM_ensemble,n_observations, n_actions).to(device)
    target_net = GaussianMixtureMLP(args.NUM_ensemble,n_observations, n_actions).to(device)

target_net.load_state_dict(policy_net.state_dict())
# set the buffer
buffer = ReplayMemory(100000)

##########################################################################################################

steps_done=0
if __name__=="__main__":
    cum_R=[]
    for i_episode in range(args.NUM_episodes):
        # Initialize the environment and get it's state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        E_count=0
        for t in count():
            steps_done+=1
            
            if args.foot_record: 
                if steps_done<20000:
                    logging.info(f"foot: {state[0,0].item()} {state[0,1].item()}")
            # select action accroding to Free energy
            if args.model=="DQN":
                action,E=select_action(policy_net,state,env,steps_done)
            elif args.model=="ensemble_DQN":
                action,E = select_action_FE(policy_net,state,args.PRINT)
            # count the explore step number
            E_count+=E
            # step forward
            observation, reward, terminated, truncated, _ = env.step(action.item())
            
            reward = torch.tensor([reward], device=device)
            done = terminated 

            if terminated:
                next_state=None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            buffer.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            # select action accroding to Free energy
            if args.model=="DQN":
                optimize_model(buffer,policy_net,optimizer,target_net,GAMMA=args.GAMMA,BATCH_SIZE=args.BATCH_SIZE) 
            elif args.model=="ensemble_DQN":
                optimize_model_ensemble(buffer,policy_net,target_net,GAMMA=args.GAMMA,BATCH_SIZE=args.BATCH_SIZE,device=device)
            # soft update th target network
            soft_update_model_weights(policy_net,target_net,args.TAU)

            if done:
                print(i_episode," step: ",t+1)
                logging.info(f" {i_episode}, step: {t+1},E: {E_count/(t+1)}")
                cum_R.append(t+1)
                ensemble=True
                if args.model=="DQN":
                    ensemble=False
                # getRM(policy_net,False,f"Q_table_best_action/best_action_{args.model}_{i_episode}_{args.file_identify}.png",ensemble)
                break

    print('Complete')
    plt.plot(cum_R)
    plt.show()
    plt.savefig(f"imgs/cumR_{args.model}_{i_episode}_{args.file_identify}.png")
    torch.save(policy_net.state_dict(),f"models_saved/{args.model}_{i_episode}_{args.file_identify}.pt")    



    