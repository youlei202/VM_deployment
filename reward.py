
# coding: utf-8

# In[1]:


import sys
import scipy.io
import numpy as np
import math
import csv
from operator import add

from gurobipy import *

# set the directory path
import os
folder_name = os.getcwd()

import time

from arrival_rate import *

epsilon = 10e-6


# In[2]:


class Problem:
    
    def __init__(self, d, P, C, c = None, p = None):
        if c == None:
            self.c = [ [2, 3, 4], 
                  [3, 1, 2],
                  [5, 2, 2] ]
        else:
            self.c = c
            
        if p == None:
            self.p = [ 10, 20, 4, 5, 6, 5, 2, 3, 4, 2, 
                  2, 2, 3, 3, 2, 2, 5, 3, 2, 3,
                  2, 1, 1, 1
                ]*2
        else:
            self.p = p
            
        self.d = d
        self.P = P
        self.C = C
        self.T = len(self.d[0])
        self.I = len(self.c[0])
        
    def solve(self):
        model = Model('Integer Programming')
        model.modelSense = GRB.MAXIMIZE
        model.setParam('OutputFlag', False) # slience output

        x = []
        for j in range(self.I):
            x_j = []
            for t in range(self.T):
                x_j.append(model.addVar(vtype = GRB.INTEGER))
            x.append(x_j)

        model.update()

        y = []
        for i in range(self.I):
            y_i = []
            for j in range(self.I):
                y_ij = []
                for t in range(self.T):
                    y_ij.append(model.addVar(vtype = GRB.INTEGER))
                y_i.append(y_ij)
            y.append(y_i)

        model.update()

        z = []
        for j in range(self.I):
            z_j = []
            for t in range(self.T):
                z_j.append(model.addVar(vtype = GRB.INTEGER))
            z.append(z_j)    
        model.update()

        eta = model.addVar(vtype = GRB.CONTINUOUS)

        for t in range(self.T):
            model.addConstr( sum(self.c[i][j]*y[i][j][t] for j in range(self.I) for i in range(self.I) ) <= self.C[t] ) 
        model.update()

        for j in range(self.I):
            model.addConstr( x[j][0] == z[j][0] )
        model.update()

        for j in range(self.I):
            for t in range(1,self.T):
                model.addConstr(x[j][t] == x[j][t-1] + sum(y[i][j][t] for i in range(self.I)) + z[j][t] )
        model.update()

        model.addConstr( sum(self.p[t]*z[j][t] for j in range(self.I) for t in range(self.T)) <= self.P )
        model.update()

        for i in range(self.I):
            for t in range(self.T):
                model.addConstr( eta <= x[i][t]/(self.d[i][t]*1.0) )
        model.update()

        model.setObjective( eta )

        model.optimize()
        
        if model.Status == GRB.OPTIMAL:
            obj_val = model.ObjVal
            x_sol = []
            for i in range(self.I):
                x_sol_i = []
                for t in range(self.T):
                    x_sol_i.append(x[i][t].X)
                x_sol.append(x_sol_i)
        else:
            obj_val = 0
            x_sol = [ [epsilon for t in range(self.T)] for i in range(self.I) ]
        return [obj_val, x_sol]


# In[447]:


# d = [ [ 1, 2, 2, 1, 2, 1, 2, 3, 4, 2, 
#       2, 1, 30, 3, 3, 2, 5, 1, 2, 3,
#       3, 3, 20, 1
#     ] for i in range(I) ]
# P = 230
# C = [ 10, 59, 29, 18, 20, 15, 2, 3, 4, 2, 
#       25, 12, 12, 31, 33, 19, 50, 1, 2, 3,
#       33, 32, 10, 10
#     ]
# print get_reward(d, P, C)


# In[74]:


# !jupyter nbconvert --to script reward.ipynb

