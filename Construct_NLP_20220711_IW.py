#!/usr/bin/env python
# coding: utf-8

# In[62]:


#import casadi
from casadi import *
import numpy as np
import math
import warnings


# In[60]:


import numpy as np

def set_initial_conditions(*varargin):
    if len(varargin) == 2:
        t0 = varargin[1-1]
        Params = varargin[2-1]
     
        #initial conditions for auxiliary state variables
        mu0 = Params.miu0 # initial mitigation rate
        s0= 0.259029014481802 # initial savings rate
     
    elif  len(varargin) == 3:
        t0 = varargin[1-1]
        x_ini = varargin[2-1]
        Params = varargin[3-1]
     
        #initial conditions for auxiliary state variables
        mu0 = x_ini(15) # initial mitigation rate == casadi var
        s0= x_ini(16) # initial savings rate == casadi var

    x0 =np.transpose([Params.T_AT0, Params.T_LO0, Params.M_AT0, Params.M_UP0, Params.M_LO0, Params.K0])

    Gross_Economic_Output0 = Params.A0*(x0[6-1]**Params.gamma)*((Params.L0/1000)**(1-Params.gamma))
    Emissions0 = 5*(Params.sigma0*(1-mu0)*Gross_Economic_Output0 + Params.EL0)
    Damages0 = 1 / (1 + (Params.a2*(x0[1-1]**Params.a3)) + (Params.a4*x0[1-1])**Params.a5)
    theta10 = Params.sigma0 * (Params.pb/(1000*Params.theta2)) * (1-Params.deltaPB)**(t0-1)
    Net_Economic_Output0 = Damages0 * (1 - theta10*(mu0**Params.theta2))*Gross_Economic_Output0
    C0 = 5*(1-s0) * Net_Economic_Output0
    result=[x0,Gross_Economic_Output0,Emissions0,Damages0,theta10,Net_Economic_Output0,C0]
    return result 


# In[59]:


def ConstructNLP(N, x0, Params):
    nx = 1+6+9+1 #number of states: time, 6 states, 9 auxilliary states, objective
    nu = 2       #number of inputs
    if len(x0) != nx:
        warnings.warn('Inconsistent initial condition x0')
        warnings.warn(str(len(x0))+' must be '+str(nx) )
        
    #u_LB = zeros(nu, N+1)
    #u_UB = ones(nu,  N+1)

    #To reproduce Nordhaus DICE2013 results, uncomment lines below    
    u_LB = [[np.zeros((1, N+1))], [np.zeros((1,N-11)), Params.optlrsav*np.ones((1, 11)), 0]]
    u_UB = [[1, 1.0*np.ones((1,27)), Params.limmiu*np.ones((1, N-27))], [np.ones((1,N+1))]]

    x_LB = np.zeros((nx, N+1))
    x_LB[4-1, :] = 1E0 #to prevent numerical problems in logarithmic function 
    x_LB[8-1:14,:] = -np.inf
    x_LB[nx-1,:] = -np.inf # no lower bound on objective
    x_UB = 1E12*np.ones((nx, N+1)) # no upper bound on states
    x_UB[end-2:end-1, :] = 1 # upper bound on shifted states (inputs)
    x_UB[2-1,:] = Params.T_AT_max # temperature constraint

    xu_LB = [[x_LB[:]], [u_LB[:]]]
    xu_UB = [[x_UB[:]], [u_UB[:]]]
    
    x           = SX.sym('x', nx, N+1) # states
    u           = SX.sym('u', nu, N+1) # inputs
    xini        = SX.sym('xini', nx, 1) # initial condition --> parameter of NLP 
    eq_con      = SX.sym('eq_con', nx, N+1) # nx * N+1 constraints for the dynamics
    
    #create a list
    x0_ini = set_initial_conditions(1, xini, Params)

    # loop over dynamics
    j=1
    for  i in range (0,N): #original i= 1
        if  i == 0:
            eq_con[:,1] = x[:,1] - xini# x(:,1) = x0       
        #should the matrix start form 0 or 1?
        # equality constraints for dynamics
        eq_con[:,i+1] = x[:,i+1] - dice_dynamics(x[:,i],u[:,i],Params)  

    # extra constraints, leaving initial states for input shift as decision variables
    eq_con_extra = [xini[1-1:end-3] - x0_ini[1-1:end-3], xini(end) - x0_ini(end)] 
    eq_con_ini = [eq_con[:], eq_con_extra[:]]

    # define the objective (Mayer term)
    #obj = ((5 * 0.016408662 * Params.sc.J*x(nx, N+1)) - 3855.106895);
    obj = 1*((x[nx-1, N+1]) ) #--> improved scaling

    # define NLPs 

# NLP for very first OCP
#nlp_ini = struct('x', [[x(:); xini(:)]; u(:)], 'f', obj, 'g', eq_con_ini(:));
    nlp_ini={}
    nlp_ini['x']=[[x[:], xini[:]],[u[:]]]
    nlp_ini['f']=obj
    nlp_ini['g']=eq_con_ini[:]


    # NLP for NMPC sequence of OCPs
    nlp={}
    nlp['x']=[x[:],u[:]]
    nlp['f']=obj
    nlp['g']=eq_con[:]
    nlp['p']=xini
    # ========================================================================
    # construct guess for states and inputs
    # =========================================================================

    u_guess = .5*np.ones((nu, N+1))
    x_guess = np.zeros((nx, N+1))
    for i in range  (1-1,N):
        if i == 1-1:
            x_guess[:,1] = x0

        x_guess[:,i+1] = dice_dynamics(x_guess[:,i],u_guess[:,i],Params)

    xu_guess = [x_guess[:], u_guess[:]]
    xu_guess_ini = [[x_guess[:], x0[:]],[u_guess[:]]]

    # ========================================================================
    # prepare output data
    # =========================================================================
    SingleOCP.nlp = nlp_ini
    SingleOCP.xu_guess = xu_guess_ini
    SingleOCP.xu_LB = xu_LB_ini
    SingleOCP.xu_UB = xu_UB_ini
    SequenceOCP.max_iter = 300

    SequenceOCP.nlp = nlp
    SequenceOCP.xu_guess = xu_guess
    SequenceOCP.xu_LB = xu_LB
    SequenceOCP.xu_UB = xu_UB
    SequenceOCP.max_iter = 300


# In[ ]:





# In[ ]:





# In[ ]:




