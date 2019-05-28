from agent_dir.agent import Agent
# import scipy
import cv2
import numpy as np
import torch
import torch.nn as nn

def prepro(o,image_size=(80,80)):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    
    """
    y = o.astype(np.uint8)
    #resized = scipy.misc.imresize(y, image_size)
    #return np.expand_dims(resized.astype(np.float32),axis=2)
    resized = cv2.resize(y, image_size, interpolation=cv2.INTER_CUBIC)
    resized = resized.astype(np.float32)/255.
    return np.swapaxes(np.swapaxes(resized, 1, 2), 0,1)


class AgentNN(torch.nn.Module):
    def __init__(self, input_dim=1, output_class=2):
        super(AgentNN,self).__init__()
        self.input_dim = input_dim
        self.output_class = output_class
        
        self.convlayers = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            # [batch, 16,20,20]
        )
        self.fully = nn.Sequential(
            # [batch, 16,20,20]
            nn.Linear(16*20*20, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, self.output_class),
            nn.Softmax(-1)
        )
        
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        out = self.convlayers(x)
        #print(out.shape)
        out = self.fully(out.view(batch_size, -1))
        #print(out.shape)
        return out
        


class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)

        if args.test_pg:
            #you can load your model here
            print('loading trained model')

        ##################
        # YOUR CODE HERE #
        ##################
        self.lastframe = prepro(self.trim(np.zeros((210, 160, 3), dtype=np.float32)))
        self.gamma = 0.99
        self.device = torch.device('cuda')
        self.agent = AgentNN(input_dim=3).to(self.device)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=1e-3)


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
        epochs = 500 
        round_episodes = 5
        self.env.seed(7373)
        
        for e in range(epochs):
            
            episode_losses = []
                        
            for i in range(round_episodes):
                #playing one game
                
                current_rewards = []
                current_logprob = []
                
                state = self.env.reset()
                self.init_game_setting()
                done = False                
                
                while(not done):
                    logp_action, action = self.make_action(state, test=False)
                    state, reward, done, info = self.env.step(action)
                    current_rewards.append(reward)
                    current_logprob.append(logp_action)
                
                ### make rewards for each action
                current_rewards_adjust = []
                littleR = 0
                for r in current_rewards[::-1]:
                    littleR = r + self.gamma*littleR
                    current_rewards_adjust.append(littleR)
                current_rewards_adjust = current_rewards_adjust[::-1]

                #episode_rewards.append(current_rewards_adjust)
                #episode_logprobs.append(current_logprob)
                            
                ### compute loss
                loss = torch.stack(current_logprob, 0) * torch.tensor(current_rewards_adjust).to(self.device)
                episode_losses.append(loss.sum())
                
            final_loss = -torch.stack(episode_losses, 0).mean()
            self.optimizer.zero_grad()
            final_loss.backward()
            self.optimizer.step()
            print(final_loss.item())
            
            
            
        


    def trim(self, screen):
        # (176,160,3)
        return screen[34:,:,:]
        
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        processed = prepro(self.trim(observation))        
        
        
        residual = processed - self.lastframe
        self.lastframe = processed
        
        #INPUT: residual (176, 160, 3)
        #OUTPUT: 0/1
        
        input_t = torch.tensor([residual]).to(self.device)
        probs = self.agent.forward(input_t)[0]
        distri = torch.distributions.Categorical(probs)
        index = distri.sample()
        
        
        #print(self.env.env.unwrapped.get_action_meanings())
        #['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
        
        if test:
            return 2 if index==0 else 3
        else:
            return torch.log(probs[index]+1e-8), (2 if index==0 else 3)
