from agent_dir.agent import Agent
import os
import math
import random
import numpy as np
import pickle
import torch
import torch.nn as nn
from collections import namedtuple
from itertools import count
from torchsummary import summary
from collections import deque


# ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
ACTION_SIZE = 4

Transition = namedtuple('Transition',
                ('state', 'action', 'next_state', 'reward')) 

#In memory state size : (84, 84, 4)
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    def push(self, *args):
        """ Saves a transition """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
        
    def __len__(self):
        return len(self.memory)
    
    
        

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
        self.fc2 = nn.Linear(512, ACTION_SIZE)
        
    def forward(self, observation):        
        observation = self.CONVS(observation)                
        observation = observation.view(-1, self.linear_input_size)        
        observation= self.fc1(observation)
        actionsQ = self.fc2(observation)
        
        return actionsQ 
        

"""
    in main:
        env = Environment(env_name, args, atari_wrapper=True)
        agent = Agent_DQN(env, args)
    
    
    for deep q learning:
        observation size is (84, 84, 4) : 4 gray-scale frames stacked
    
    for the atari game breakout:
        use self.env.env.unwrapped.get_action_meanings() to get the action space
        action space : ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
        action size : 4
"""


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
            class Agent(object):
                def __init__(self, env):
                    self.env = env
                    
            self.env means Environment, containing 6 functions:
                1. seed(seed)
                2. reset()
                3. step(action)
                4. get_action_space()
                5. get_observation_space()
                6. get_random_action()        
                
        """
        
        super(Agent_DQN,self).__init__(env)
        self.DQN_INPUT_SIZE = (4, 84, 84)
        self.BATCH_SIZE = 32
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_DECAY = 100000
        self.EPS_END = 0.025
        
        self.steps_done = 0
        self.Loss_hist = []
        self.Reward_hist = []
        
        self.device = torch.device('cuda')        
        self.Q_policy = DQN(self.DQN_INPUT_SIZE).to(self.device)     
        self.Q_target = DQN(self.DQN_INPUT_SIZE).to(self.device)
        self.Q_target.load_state_dict(self.Q_policy.state_dict())
        self.Q_target.eval()
        self.memory = ReplayMemory(10000)
        self.RewardQueue = deque(maxlen=30)
        self.AverageReward_hist = []
        
        self.optimizer = torch.optim.Adam(self.Q_policy.parameters(), lr=1e-4)    
        self.MSE_loss = nn.MSELoss().to(self.device)
        """------------------------------------------------------------------"""
        
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')   
            
            
        print("Using Device : {}".format(self.device))
        print('---------- Networks architecture -------------')
        summary(self.Q_policy, (self.DQN_INPUT_SIZE))
        print('----------------------------------------------')

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################self.noise_dim,
        pass
    
    def train(self):
        
        # observation shape : (84, 84, 4)
        # -> transpose((2, 0, 1))
        # -> shape : (4, 84, 84)
        
        def save(i_episode, dump_data):
            save_dir = "Q_saved_base"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(self.Q_policy.state_dict(), os.path.join(save_dir, str(i_episode) + "_Q.pkl"))
            with open(os.path.join(save_dir, '{}.pkl'.format(i_episode)), 'wb') as f:
                pickle.dump(dump_data, f)
        
        #######################################################################
        #                       MAIN TRAINING LOOP                            # 
        #######################################################################
        for i_episode in count():
            state = self.env.reset()
            REWARD = 0
            for t in count():
                # in make_action self.step_done += 1
                # IMPORTANT! : make_action receive size (84,84,4)
                action = self.make_action(state)
                next_state, reward, done, _ = self.env.step(action)   
                self.memory.push(state, [action], next_state, [int(reward)])
                state = next_state       
                
                if self.steps_done % 4 ==0:
                    self.optimize_model()
                    
                if self.steps_done % 1000 == 0:
                    print("Q_target <- Q_policy")
                    self.Q_target.load_state_dict(self.Q_policy.state_dict())                    
                    
                REWARD = REWARD + reward                
                if done:           
                    self.RewardQueue.append(REWARD)
                    average_reward = sum(self.RewardQueue) / len(self.RewardQueue)
                    self.AverageReward_hist.append(average_reward)
                    print("episode : {}, step : {}, average_reward:{}".format(i_episode, self.steps_done, average_reward))
                    break                        
            

            
            if i_episode % 1000 == 0:
                dump_data = [self.steps_done, i_episode, self.AverageReward_hist]
                print("episode : ", i_episode)
                print("saving model...")
                save(i_episode, dump_data)
            
            
                
    def make_action(self, observation, test=True):
        """
            Return predicted action of your agent
    
            Input:
                observation: np.array
                    stack 4 last preprocessed frames, shape: (84, 84, 4)
    
            Return:
                action: int
                    the predicted action from trained model
        """
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) *\
                            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        
        # epsilon greedy
        if np.random.rand() < eps_threshold:
            """ random """
            return self.env.get_random_action()
        
        
        else:
            """ greedy """
            # input is (84, 84, 4)
            # permute -> (4, 84, 84)
            # unsqueeze -> (1, 4, 84, 84)
            
            with torch.no_grad():
                observation = torch.FloatTensor(observation).permute((2,0,1)).unsqueeze(0).to(self.device)
                actionsQ = self.Q_policy(observation)
                return torch.argmax(actionsQ).item()
            
            
    def optimize_model(self):
        # there should be enough data in memory
        
        if len(self.memory) < self.BATCH_SIZE:
            return 
        
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        # zip(*[('a', 1), ('b', 2), ('c', 3), ('d', 4)])
        # ->[('a', 'b', 'c', 'd'), (1, 2, 3, 4)]
        # converts batch-array of Transitions to Transition of batch-arrays 

        def to_tuple_of_tensor(t):            
            return(tuple(torch.Tensor(e).unsqueeze(0) for e in list(t)))
        
        # 1. batch.next_state
        next_state_batch = torch.cat(to_tuple_of_tensor(batch.next_state)).float().to(self.device)
        next_state_batch = next_state_batch.permute((0, 3, 1, 2)) # to BCHW
        
        # 2. batch.state
        state_batch = torch.cat(to_tuple_of_tensor(batch.state)).float().to(self.device)
        state_batch = state_batch.permute((0, 3, 1, 2)) # to BCHW
        
        # to long is for gather
        # 3. batch.action
        action_batch = torch.cat(to_tuple_of_tensor(batch.action)).to(self.device).long()
        
        # 4. batch.reward
        reward_batch = torch.cat(to_tuple_of_tensor(batch.reward)).to(self.device)
        
        
        #debug
        """
        if self.steps_done == 200:
            print("saved")
            with open('actions.pkl', 'wb') as f:
                    pickle.dump(action_batch, f)
            with open('transistions.pkl', 'wb') as f:
                    pickle.dump(batch, f)        
        """
        
        Qvalue_t0 = self.Q_policy(state_batch).gather(1, index=action_batch)
        Qvalue_t1 = self.Q_target(next_state_batch).max(1)[0].unsqueeze(1).detach()
        expected_Qvalue = (Qvalue_t1*self.GAMMA) + reward_batch
        
        loss = self.MSE_loss(Qvalue_t0, expected_Qvalue)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.Q_policy.parameters():
            param.grad.data.clamp_(-1, 1)        
        self.optimizer.step()
    