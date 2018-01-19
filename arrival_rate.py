
# coding: utf-8

# In[134]:


from math import *
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

epsilon = 10e-6


# In[205]:


class ArrivalController:
    
    period_num = 8;
    slot_len = 3
    slot_num = period_num * slot_len
    episode_num = 100;
    
    def __init__(self, 
                 period_alpha = None , 
                 slot_alpha = None, 
                 latency_list = None
                ):
        if period_alpha == None:
            self.period_alpha = np.array([0.1, 0.5, 1.7, 1.85, 1.75, 1.4, 0.1, 0.2]) * 100
        else:
            self.period_alpha = period_alpha
            
        if slot_alpha == None:
            self.slot_alpha = np.array([1.0, 1.0, 1.0]) * 10
        else:
            self.slot_alpha = slot_alpha
            
        if latency_list == None:
            self.latency_list = [1,1,1]
        else:
            self.latency_list = latency_list
                
        
        def get_delta(object):
            dirchlet_list = self.get_dirchlet_list('period')
            delta = ArrivalController.period_num * dirchlet_list
            return delta
        
        def get_eta(object):
            eta = np.empty([1,0])
            for i in range(ArrivalController.period_num):
                dirchlet_list = ArrivalController.slot_len * self.get_dirchlet_list('slot')
                eta = np.append(eta,dirchlet_list)
            eta.shape = (ArrivalController.period_num, ArrivalController.slot_len)
            return eta
        
        self.delta = get_delta(self)
        self.eta = get_eta(self)
    
    def period_index(self, time_index):
        return int(floor((time_index / ArrivalController.slot_len) % ArrivalController.period_num ))
    
    def slot_index(self, time_index):
        return int(floor(time_index % ArrivalController.slot_len))
    
    def get_dirchlet_list(self, name):
        if name == 'period':
            return np.random.dirichlet(self.period_alpha).transpose()
        elif name == 'slot':
            return np.random.dirichlet(self.slot_alpha).transpose()
        else:
            return -1
    
    def set_latency(self,latency_list):
        self.latency_list = latency_list
    
    def avg_rate(self):
        return [30/(1.0*x) for x in self.latency_list]
    
    def get_lambda(self):
        lam = []
        for i in range(len(self.latency_list)):
            lam_i = []
            rate_i = self.avg_rate()[i]
            for t in range(self.slot_num):
                lam_it = ( rate_i * self.delta[self.period_index(t)]
                     * self.eta[self.period_index(t), self.slot_index(t)] )
                lam_i.append(lam_it)
            lam.append(lam_i)
        return lam
    
    def get_arrival_users(self):
        user_num_list = []
        lam = self.get_lambda()
        for i in range(len(lam)):
            user_num_i = []
            for t in range(len(lam[0])):
                user_num_it = np.random.poisson(lam[i][t])
                if user_num_it == 0:
                    user_num_it = epsilon
                user_num_i.append(user_num_it)
            user_num_list.append(user_num_i)
        return user_num_list
    
    


# In[210]:


# period_alpha = np.array([0.1, 0.5, 1.7, 1.85, 1.75, 1.4, 0.1, 0.2]) * 100
# slot_alpha = np.array([1.0, 1.0, 1.0]) * 10
# latency_list = [10,12,5]
# controller = ArrivalController()
# np.array(controller.get_arrival_users())


# In[209]:


# !jupyter nbconvert --to script arrival_rate.ipynb

