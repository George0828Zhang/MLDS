import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import random

from agent_dir.agent import Agent
from environment import Environment

use_cuda = torch.cuda.is_available()

ACTION_SIZE = 4

class DQN(nn.Module):    
    #dqn : input_size is (batch_size, 4, 84, 84) in CHW format
    def __init__(self, input_size=(4, 84, 84), action_size = ACTION_SIZE):
        super(DQN,self).__init__()
        (self.C, self.H, self.W)= input_size
        self.action_size = action_size
        self.CONVS = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),            
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1        
        convH = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.H, 8, 4), 4, 2), 3, 1)
        convW = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.W, 8, 4), 4, 2), 3, 1)
        
        self.linear_input_size = convW * convH * 64       
        self.fc1 = nn.Linear(self.linear_input_size, 512)
        self.relu = nn.ReLU();
        self.fc2 = nn.Linear(512, ACTION_SIZE)
        
    def forward(self, observation):        
        observation = self.CONVS(observation)                
        observation = observation.view(-1, self.linear_input_size)        
        observation= self.fc1(observation)
        observation = self.relu(observation)
        actionsQ = self.fc2(observation)       
        return actionsQ 
        

class Agent_DQN(Agent):
    def __init__(self, env, args):
        self.env = env
        self.input_channels = 1
        self.num_actions = 3#self.env.action_space.n
        # TODO:
        # Initialize your replay buffer
        #states_buffer, actions_buffer, reward_buffers = [], [], []
        self.capacity = 2000;
        self.buffer = []
        self.DQN_INPUT_SIZE = (4, 84, 84)
        # build target, online network
        self.target_net = DQN(self.DQN_INPUT_SIZE)
        self.target_net = self.target_net.cuda() if use_cuda else self.target_net
        self.online_net = DQN(self.DQN_INPUT_SIZE) 
        self.online_net = self.online_net.cuda() if use_cuda else self.online_net

        self.EPS_START = 1
        self.EPS_DECAY = 1000
        self.EPS_END = 0.02
        
        self.steps_done = 0
        
        if args.test_dqn:
            self.load('dqn')
        
        # discounted reward
        self.GAMMA = 0.99 
        #self.GAMMA = 0.75
        #self.GAMMA = 0.5
        #self.GAMMA = 0.25
        
        # training hyperparameters
        self.train_freq = 4 # frequency to train the online network
        self.learning_start = 10000 # before we start to update our network, we wait a few steps first to fill the replay.
        self.batch_size = 32
        self.num_timesteps = 3000000 # total training steps
        self.display_freq = 30 # frequency to display training progress
        self.save_freq = 200000 # frequency to save the model
        self.target_update_freq = 1000 # frequency to update target network

        # optimizer
        self.optimizer = optim.RMSprop(self.online_net.parameters(), lr=1e-4)

        self.steps = 0 # num. of passed steps. this may be useful in controlling exploration
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.total_steps = []
        self.avg_rewards = []


    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.online_net.state_dict(), save_path + '_online.cpt')
        torch.save(self.target_net.state_dict(), save_path + '_target.cpt')
        torch.save({'steps':self.total_steps, 'avg_rewards':self.avg_rewards}, 'DQNGraphRequired')

    def load(self, load_path):
        print('load model from', load_path)
        if use_cuda:
            self.online_net.load_state_dict(torch.load(load_path + '_online.cpt'))
            self.target_net.load_state_dict(torch.load(load_path + '_target.cpt'))
        else:
            self.online_net.load_state_dict(torch.load(load_path + '_online.cpt', map_location=lambda storage, loc: storage))
            self.target_net.load_state_dict(torch.load(load_path + '_target.cpt', map_location=lambda storage, loc: storage))

    def init_game_setting(self):
        # we don't need init_game_setting in DQN
        pass
    
    def make_action(self, state, test=False):
        # TODO:
        # At first, you decide whether you want to explore the environemnt

        # TODO:
        # if explore, you randomly samples one action
        # else, use your model to predict action
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) *\
                            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        
        with torch.no_grad():
            #state = state).float().unsqueeze(0)
            if(test==True):
                probs = self.online_net(torch.tensor(state, device=self.device).unsqueeze(0).transpose(3,2).transpose(2,1)).squeeze(0)
            else:
                probs = self.online_net(torch.tensor(state, device=self.device)).squeeze(0)

            if(random.random() < 1 - eps_threshold):
                value, index = torch.max(probs, 0)
                action = index.item()
            else:
                action = random.randint(0, self.num_actions - 1)

        return action

    def update(self):
        # TODO:
        # To update model, we sample some stored experiences as training examples.
        batch = random.sample(self.buffer, self.batch_size)
        # TODO:
        # Compute Q(s_t, a) with your model.
        state_batch = np.array(batch)[:,0].tolist();
        state_batch = torch.cat(state_batch, 0);
        
        action_batch = torch.tensor(np.array(batch)[:,1].tolist(), device = self.device);
        
        reward_batch = torch.tensor(np.array(batch)[:,2].tolist(), device = self.device);
        
        next_state_batch = np.array(batch)[:,3].tolist();
                                    
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          next_state_batch)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in next_state_batch
                                                if s is not None])
        
        temp = torch.tensor([[i] for i in action_batch], device = self.device);
        state_action_values = self.online_net(state_batch).gather(1, temp);  
        
        
        with torch.no_grad():
            # TODO:
            # Compute Q(s_{t+1}, a) for all next states.
            # Since we do not want to backprop through the expected action values,
            # use torch.no_grad() to stop the gradient from Q(s_{t+1}, a)
            
            # basic deep q
            #next_state_values = torch.zeros(self.batch_size, device=self.device)
            #next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
            
            # double deep q
            next_state_values = torch.zeros(self.batch_size, device=self.device)
            online_net_actions = self.online_net(non_final_next_states).max(1)[1]
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, index=online_net_actions)
            

        # TODO:
        # Compute the expected Q values: rewards + gamma * max(Q(s_{t+1}, a))
        # You should carefully deal with gamma * max(Q(s_{t+1}, a)) when it is the terminal state.

        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        
        # TODO:
        # Compute temporal difference loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self):
        episodes_done_num = 0 # passed episodes
        total_reward = 0 # compute average reward
        loss = 0 
        while(True):
            state = self.env.reset()
            # State: (80,80,4) --> (1,4,80,80)
            state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0)
            state = state.cuda() if use_cuda else state
            
            done = False
            
            
            while(not done):
                # select and perform action
                action = self.make_action(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward


                # process new state
                next_state = torch.from_numpy(next_state).permute(2,0,1).unsqueeze(0)
                next_state = next_state.cuda() if use_cuda else next_state
                if done:
                    next_state = None

                # TODO:
                # store the transition in memory
                self.buffer.append([state, action, reward, next_state])
                
                
                if(len(self.buffer) > self.capacity):
                    self.buffer = self.buffer[-self.capacity:]
                
                # move to the next state
                state = next_state

                # Perform one step of the optimization
                if self.steps > self.learning_start and self.steps % self.train_freq == 0:
                    loss = self.update()

                # update target network
                if self.steps > self.learning_start and self.steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())

                # save the model
                if self.steps % self.save_freq == 0:
                    self.save('dqn')

                self.steps += 1
            
            
            
            #print(total_reward)
            
            if episodes_done_num % self.display_freq == 0:
                print('Episode: %d | Steps: %d/%d | Avg reward: %f | loss: %f '%
                        (episodes_done_num, self.steps, self.num_timesteps, total_reward / self.display_freq, loss))
                self.avg_rewards.append(total_reward)
                total_reward = 0

            episodes_done_num += 1
            
            self.total_steps.append(self.steps)
            
            
            if self.steps > self.num_timesteps:
                break
        self.save('dqn')
