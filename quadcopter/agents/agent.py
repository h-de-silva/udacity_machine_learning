# TODO: your agent here! 

import numpy as np
from task import Task
import copy
import pdb

class Finite_Difference_Agent():
    def __init__(self, task):
        # Task (environment) information
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low
        ##### Add
        
        self.alpha_w = 0.3 #0.1
        self.epsilon_w = 0.1  #0.01
        self.is_fd_appr = False
        self.episode = 1
        ########
        
        
        temp = 20.0*np.array([[0., 0., 0., 0.],[0.,0.,0.,0.], [1.,1.,1.,1.]] )
        self.w = np.tile(temp,(6,1))
        
        
        self.score = None 
        ############
        
        # Score tracker and learning parameters
        self.best_w = None
        self.best_score = -np.inf
        self.noise_scale = 0.1

        # Episode variables
        self.reset_episode()

    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        self.episode = self.episode + 1
        state = self.task.reset()
        return state

    def step(self, reward, done):
        # Save experience / reward
        
        self.count += 1
        
        self.total_reward += reward
        
        if done and self.is_fd_appr == False: 
            self.learn_grad_desc()

    def act(self, state):
        # Choose action based on given state and policy 
        self.gamma =0
        action = np.dot(state, self.w) 
        
        target_pose = np.tile(np.concatenate([self.task.target_pos,np.array([0.,0.,0.])],axis=None),3)
        
        action = np.dot(target_pose-state, self.w)  # simple linear policy
        
         
            
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
        
        #weight = self.w.copy()
        epsilon = self.epsilon_w
        n_rows = self.state_size 
        n_cols = self.action_size
        delta_reward = np.zeros(shape = (n_rows, n_cols) )
        
        ##########################
        
        ######################################################
        
        temp = epsilon*np.array([[0., 0., 0., 0.],[0.,0.,0.,0.], [1.,1.,1.,1.]] )
        dw = np.tile(temp,(6,1))
        
        weight = self.w+ dw 
        
        
        target_pos = self.task.target_pos
        new_task = Task(target_pos=target_pos)
        new_agent = Finite_Difference_Agent(new_task) 
        new_agent.is_fd_appr = True

        state = new_agent.reset_episode()
        new_agent.w = weight.copy()
        while True: 
            new_agent.current_state = state
            action = new_agent.act(state) 


            next_state, reward, done = new_task.step(action) 
                            

            new_agent.step(reward, done)
            state = next_state
            if done:
                break

    
        new_score = new_agent.total_reward / float(new_agent.count) if new_agent.count else 0.0
                        
        
        Jw_dw = (new_score - self.score)/epsilon
        
        
        del new_task
        del new_agent        
        
        
        return [Jw_dw] 
    
    
    
    def learn_grad_desc(self):
        # Learn by random policy search, using a reward-based score
        if self.score is not None: 
            self.prev_score = self.score 
        else: 
            self.prev_score = np.inf
        self.score = self.total_reward / float(self.count) if self.count else 0.0 
        
        grad = self.get_gradient()
        
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_w = self.w
            self.best_grad = grad
            self.noise_scale = max(0.5 * self.noise_scale, 0.01)
        else:
            #self.w = self.best_w
            self.noise_scale = min(2.0 * self.noise_scale, 3.2)
        
        #temp = [self.alpha*i for i in grad ]
        temp = grad[0]*np.array([[0., 0., 0., 0.],[0.,0.,0.,0.], [1.,1.,1.,1.]] )
        temp = np.tile(temp,(6,1))
        
        
        temp = self.alpha_w*temp 
        
        #print("grad = ", grad)
        
        #print("w = ",self.w[2][3])
        
        self.w = self.w + temp
        
       

'''        
    def get_gradient(self):
        
        #weight = self.w.copy()
        epsilon = self.epsilon
        n_rows = self.state_size 
        n_cols = self.action_size
        delta_reward = np.zeros(shape = (n_rows, n_cols) )
        
        ##########################
        
        ######################################################
        
        
        for row in range(n_rows):
            for col in range(n_cols):
                
                weight = copy.copy(self.w) 
                if weight[row][col]>10.0:   
                    weight[row][col] = weight[row][col]+epsilon
                    #copy_self = copy.copy(self)
                    #copy_task = copy.copy(self.task)
                    #copy_self.w = weight 
                    target_pos = np.array([0., 0., 10.])
                    new_task = Task(target_pos=target_pos)
                    new_agent = Finite_Difference_Agent(new_task) 
                    new_agent.is_fd_appr = True

                    new_score =[]

                    state = new_agent.reset_episode()
                    new_agent.w = weight.copy()
                    while True: 
                        new_agent.current_state = state
                        action = new_agent.act(state) 


                        next_state, reward, done = new_task.step(action) 
                            #print(new_agent.count, done)

                        new_agent.step(reward, done)
                        state = next_state
                        if done:
                            break


                    new_score = new_agent.total_reward / float(new_agent.count) if new_agent.count else 0.0
                        #new_score.append(new_agent.total_reward / float(new_agent.count) if new_agent.count else 0.0)
                    #new_score = np.mean(new_score)

                    delta_reward[row][col] = (new_score - self.score)/epsilon
                    #delta_reward[row][col] = (new_score - old_score)/epsilon

                    #del copy_self
                    #del copy_task
                    del new_task
                    del new_agent
                       
        #print(delta_reward)
        return delta_reward
        
'''