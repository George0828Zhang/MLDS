from agent_dir.agent import Agent
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary
from tqdm import tqdm

class DQN(nn.Module):
    def __init__(self, input_size):
        super(DQN,self).__init__()
        self.input_size = input_size
        
        self.fully = nn.Sequential(
            nn.Linear(input_size[0]*input_size[1]*input_size[2] + 1, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )
        
    def forward(self, x, a):
        batch_size = x.shape[0]

        x = torch.cat((x.view(batch_size, -1), a), 1)
        
        out = self.fully(x)
        return out

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')

        ##################
        # YOUR CODE HERE #
        ##################
        input_size = (84,84,4)
        self.gamma = 0.99
        self.eps_greedy = 1. # 1. to full random
        self.device = torch.device('cuda')
        self.dqnn = DQN(input_size).to(self.device)
        self.dqnn_hat = DQN(input_size).to(self.device)
        self.optimizer = torch.optim.RMSprop(self.dqnn.parameters(), lr=5e-3)
        #['NOOP', 'FIRE', 'RIGHT', 'LEFT']
        #print(self.env.env.unwrapped.get_action_meanings())
        self.actions = torch.FloatTensor([0,1,2,3]).view(-1, 1).to(self.device)


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass


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
        ##################
        # YOUR CODE HERE #
        ##################
        #print(self.actions)
        
        # epsilon greedy
        if np.random.rand() < self.eps_greedy:
            """ random """
            return self.env.get_random_action()
        else:
            """ greedy """
            data_x = torch.FloatTensor(observation)
            data_x = data_x.view(1, 84, 84, 4).repeat(len(self.actions), 1, 1, 1).to(self.device)

            q_values = self.dqnn(data_x, self.actions)

            return torch.argmax(q_values).item()

