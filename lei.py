
# coding: utf-8

# In[1]:


import numpy as np
import math
from arrival_rate import *
from reward import *


# In[2]:


from tensorforce.agents import PPOAgent

# Create a Proximal Policy Optimization agent
agent = PPOAgent(
        states_spec=dict(type='float', shape=(3)),
        actions_spec=dict(type='float', shape=(1 + ArrivalController.slot_num)),
        network_spec=[
            dict(type='dense', size=64),
            dict(type='dense', size=64)
            ],
        batch_size=1000,
        step_optimizer=dict(
            type='adam',
            learning_rate=1e-4
            )
        )


# In[50]:


class MyClient:
    
    total_budget = 500
    
    def __init__(self):
        self.latency = [1, 1, 1]
        
        self.controller = ArrivalController(latency_list = self.latency)
    
        self.purchase_budget = 140
        self.transfer_budget = [ 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 
                                 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
                                 15, 15, 15, 15
                               ]

        # We must have purchase_budget + transfer_budget <= total_budget !
        
    def get_state(self):
        return self.latency
    
    def execute(self, action): # action is in the exactly same format as state
        if sum(action) > MyClient.total_budget:
            self.purchase_budget = 0
            self.transfer_budget = [0 for i in range(ArrivalController.slot_num)]
        self.purchase_budget = action[0]
        self.transfer_budget = action[1:ArrivalController.slot_num+1]
    
        self.controller.set_latency(self.latency)
        
        d = self.controller.get_arrival_users()
        P = self.purchase_budget
        C = self.transfer_budget
    
        problem = Problem(d,P,C)
        result = problem.solve()
        
        reward = result[0]
        x = result[1]
        self.latency = map(lambda x: sum(x)/ArrivalController.slot_num,
                           np.divide( np.array(d), np.array(x) ) )   
        return reward


# In[124]:


# Get new data from somewhere, e.g. a client to a web app
client = MyClient()

# Poll new state from client
state = client.get_state()

# Get prediction from agent, execute
# action = agent.act(state)
action = [140, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 
               15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
               15, 15, 15, 15
         ]
reward = client.execute(action)
print client.get_state(), reward
# Add experience, agent automatically updates model according to batch size
agent.observe(reward=reward, terminal=False)



# In[1]:


# !jupyter nbconvert --to script lei.ipynb

