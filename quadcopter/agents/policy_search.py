import numpy as np
from task import Task
import copy
import pdb

class PolicySearch_Agent():
    def __init__(self, task):
        # Task (environment) information
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low
        ##### Add
        self.current_state = None
        self.alpha = 0.2
        self.epsilon = 0.01
        ########
        self.w = np.random.normal(
            size=(self.state_size, self.action_size),  # weights for simple linear policy: state_space x action_space
            scale=(self.action_range / (2 * self.state_size))) # start producing actions in a decent range

        # Score tracker and learning parameters
        self.best_w = None
        self.best_score = -np.inf
        self.noise_scale = 0.1

        # Episode variables
        self.reset_episode()

    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        state = self.task.reset()
        return state

    def step(self, reward, done):
        # Save experience / reward
        self.total_reward += reward
        self.count += 1

        # Learn, if at end of episode
        if done:
            self.learn()
            

    def act(self, state):
        # Choose action based on given state and policy
        #pdb.set_trace()
        action = np.dot(state, self.w)  # simple linear policy
        return action

    def learn(self):
        # Learn by random policy search, using a reward-based score
        self.score = self.total_reward / float(self.count) if self.count else 0.0
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_w = self.w
            self.noise_scale = max(0.5 * self.noise_scale, 0.01)
        else:
            self.w = self.best_w
            self.noise_scale = min(2.0 * self.noise_scale, 3.2)
        self.w = self.w + self.noise_scale * np.random.normal(size=self.w.shape)  # equal noise in all directions
        
        
    ##################### Changing 
    
    def get_gradient(self):
        delta_reward = []
        weight = self.w.copy()
        epsilon = self.epsilon
        for i in range(len(weight)):
            weight[i] = weight[i]+epsilon
            copy_self = copy.copy(self)
            copy_task = copy.copy(self.task)
            copy_self.w = weight            
            action = copy_self.act(copy_self.current_state)
            next_state, reward, done = copy_task.step(action) 
            delta_reward.append((self.total_reward - reward)/epsilon) 
            del copy_self
            del copy_task
        return delta_reward
        
    def learn_grad_desc(self):
        # Learn by random policy search, using a reward-based score
        self.score = self.total_reward / float(self.count) if self.count else 0.0
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_w = self.w
            self.noise_scale = max(0.5 * self.noise_scale, 0.01)
        else:
            #self.w = self.best_w
            self.noise_scale = min(2.0 * self.noise_scale, 3.2)
        grad = self.get_gradient()
        temp = [self.alpha*i for i in grad ]
        self.w = self.w + temp
        self.w = self.w + self.noise_scale * np.random.normal(size=self.w.shape)    
        