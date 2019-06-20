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

class A2C(nn.Module):    
    #dqn : input_size is (batch_size, 4, 84, 84) in CHW format
    def __init__(self, input_size=(4, 84, 84), action_size = ACTION_SIZE):
        super(A2C,self).__init__()
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
        self.actor = nn.Sequential(
                        nn.Linear(self.linear_input_size, ACTION_SIZE),
                        nn.Softmax(-1),
                        )
        self.critic = nn.Linear(self.linear_input_size, 1)
        
    def forward(self, observation):        
        a = self.CONVS(observation)                
        a = a.view(-1, self.linear_input_size)        
        value = self.critic(a)
        action_prob = self.actor(a)
        return value, action_prob 
        

class Agent_A2C(Agent):
    def __init__(self, env, args):
        self.env = env
        self.input_channels = 1
        self.num_actions = 3#self.env.action_space.n
        
        self.steps_done = 0
        
        if args.test_dqn:
            self.load('dqn')
        
        # discounted reward
        self.GAMMA = 0.99 
        #self.GAMMA = 0.75
        #self.GAMMA = 0.5
        #self.GAMMA = 0.25
        
        # training hyperparameters
        self.update_freq = 4 
        self.batch_size = 32
        self.num_timesteps = 300000000 # total training steps
        self.display_freq = 30 # frequency to display training progress
        self.save_freq = 200000 # frequency to save the model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = A2C().to(self.device)
        
        # optimizer
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=1e-4)

        self.steps = 0 # num. of passed steps. this may be useful in controlling exploration
        
        
        
        self.total_steps = []
        self.avg_rewards = []
        self.loss = []


    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.model.state_dict(), save_path + '.cpt')
        torch.save({'steps':self.total_steps, 'avg_rewards':self.avg_rewards, 'loss':self.loss}, 'A2CGraphRequired')

    def load(self, load_path):
        print('load model from', load_path)
        if use_cuda:
            self.model.load_state_dict(torch.load(load_path + '.cpt'))
        else:
            self.model.load_state_dict(torch.load(load_path + '.cpt', map_location=lambda storage, loc: storage))

    def init_game_setting(self):
        # we don't need init_game_setting in DQN
        pass
    
    def make_action(self, state, test=False):
        with torch.no_grad():
            value, action_prob = self.model(state)
            
            if(test == True):
                action = action_prob.argmax(-1);
            else:               
                m = torch.distributions.Categorical(action_prob)
                action = m.sample()
            
        return value.item(), action.item()

    def update(self, states, actions, rewards, dones):
        R = []
        if(dones[-1] == True):
            next_return = 0;
        else:
            value, action = self.make_action(states[-1])
            #append next value
            next_return = value;
        
        R = []
        
        #r(s,a) + gamma * V(next_s)
        for r,done in zip(rewards[::-1],dones[::-1]):
            if(done == True):
                current_return = r + 0;
            else:
                current_return = r + self.GAMMA * next_return;
            R.append(current_return)
            next_return = current_return
            
        R.reverse()
        
        
        values, action_probs = self.model(torch.stack(states, 0).to(self.device).squeeze(1))
        action_log_probs = action_probs.log()
        
        chosen_action_log_probs = action_log_probs.gather(1, torch.tensor(actions).unsqueeze(1).to(self.device))
        
        advantages = torch.tensor(R).to(self.device).float() - values
        
        self.optimizer.zero_grad()
        #gradient ascent
        loss = -1 * (chosen_action_log_probs * advantages).mean()
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
            
            states = []
            actions = []
            rewards = []
            dones = []
            while(not done):
                # select and perform action
                
                #value is not important here
                _ , action = self.make_action(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                
                # process new state
                next_state = torch.from_numpy(next_state).permute(2,0,1).unsqueeze(0)
                next_state = next_state.cuda() if use_cuda else next_state
                
                    
                
                if done:
                    next_state = None
                # move to the next state
                state = next_state
                
                if (self.steps + 1) % self.update_freq == 0:
                    self.update(states, actions, rewards, dones)
                
                # save the model
                if self.steps % self.save_freq == 0:
                    self.save('a2c')

                self.steps += 1
            
            
            
            #print(total_reward)
            
            if episodes_done_num % self.display_freq == 0:
                print('Episode: %d | Steps: %d/%d | Avg reward: %f | loss: %f '%
                        (episodes_done_num, self.steps, self.num_timesteps, total_reward / self.display_freq, loss))
                self.avg_rewards.append(total_reward/self.display_freq)
                self.loss.append(loss)
                total_reward = 0

            episodes_done_num += 1
            
            self.total_steps.append(self.steps)
            
            
            if self.steps > self.num_timesteps:
                break
        self.save('a2c')
