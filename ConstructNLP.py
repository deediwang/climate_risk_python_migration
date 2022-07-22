#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
import numpy as np
def ConstructNLP(N, x0, Params):
    
    #Construction of NLPs to be solved at each sampling instant
    #Preliminary functions: set_initial_conditions
    #Input: prediction horizon, initial conditions, initial parameters
    #Output: struct with OCP definition for very first time instant, struct with OCP definition for MPC solution
    #Author: Ivan Wang 20220715, Xiaoman Zhang 20220721
    
    def SX_multiply(A, B):
        res = []
        n = len(A[0])
        m = len(A)
        for j in range(m):
            temp = 0
            for i in range(n):
                temp+=B[i]*A[j][i]
            res.append(temp)
        return res
    
    nx = 1+6+9+1 #number of states: time, 6 states, 9 auxilliary states, objective
    nu = 2       #number of inputs
    if len(x0) != nx:
        warnings.warn('Inconsistent initial condition x0')
        print('Inconsistent initial condition x0')
        print('Length of X0 must be '+str(nx))

    #To reproduce Nordhaus DICE2013 results, uncomment lines below    
    u_LB = [[0]*(N+1), [0]*(N-11)+[Params.optlrsav]*11+ [0]]
    u_UB = [[1]+[1]*27+[Params.limmiu]*(N-27), [1]*(N+1)]

    x_LB = np.zeros((nx, N+1))
    x_LB[3, :] = 1 #to prevent numerical problems in logarithmic function 
    x_LB[7:14,:] = -np.inf
    x_LB[nx-1,:] = -np.inf # no lower bound on objective
    x_UB = 1E12*np.ones((nx, N+1)) # no upper bound on states
    x_UB[nx-3:nx-1, :] = 1 # upper bound on shifted states (inputs)
    x_UB[1,:] = Params.T_AT_max # temperature constraint
    
    x_LB_flat = []
    for i in range(len(x_LB.T)):
        x_LB_flat+=list(x_LB.T[0])
    x_UB_flat = []
    for i in range(len(x_UB.T)):
        x_UB_flat+=list(x_UB.T[0])

    u_LB_flat = []
    for i in range(len(np.array(u_LB).T)):
        u_LB_flat+=list(np.array(u_LB).T[0])
    u_UB_flat = []
    for i in range(len(np.array(u_UB).T)):
        u_UB_flat+=list(np.array(u_UB).T[0])
        
    xu_LB = x_LB_flat+u_LB_flat
    xu_UB = x_UB_flat+u_UB_flat
    
    # constraints on initial state (first OCP)
    x_LB_ini = np.zeros((nx, N+2))
    x_LB_ini[3, :] = 1E0; # to prevent numerical problems in logarithmic function 
    x_LB_ini[7:14,:] = -np.inf
    x_LB_ini[nx-1,:] = -np.inf # no lower bound on objective
    x_LB_ini[nx-3:nx-1, N+1] = 0
    x_UB_ini = 1E12*np.ones((nx, N+2)) # no upper bound on states
    x_UB_ini[1,:] = Params.T_AT_max # temperature constraint

    # To reproduce Nordhaus DICE2013 results, uncomment lines below
    x_UB_ini[nx-3:nx-1, :] = Params.limmiu #upper bound on shifted states, i.e. inputs
    x_UB_ini[nx-3, N+1] = Params.miu0
    x_UB_ini[nx-2, N+1] = 1  #initial input constraints in Nordhaus' code, need to reproduce Nordhaus' results
    
    x_LB_ini_flat = []
    for i in range(len(x_LB_ini.T)):
        x_LB_ini_flat+=list(x_LB_ini.T[0])
    x_UB_ini_flat = []
    for i in range(len(x_UB_ini.T)):
        x_UB_ini_flat+=list(x_UB_ini.T[0])
    
    xu_LB_ini = x_LB_ini_flat+u_LB_flat
    xu_UB_ini = x_UB_ini_flat+u_UB_flat
    
    # ========================================================================
    # Define NLP
    # =========================================================================
    # allocate CASADI variables
    x           = SX.sym('x', nx, N+1) # states
    u           = SX.sym('u', nu, N+1) # inputs
    xini        = SX.sym('xini', nx, 1) # initial condition --> parameter of NLP 
    eq_con      = SX.sym('eq_con', nx, N+1) # nx * N+1 constraints for the dynamics
    
    #create a list
    x0_ini = set_initial_conditions(Params, 1, xini, True)
    
    for  i in range (N): #original i= 1
        if  i == 0:
            eq_con[:,0] = x[:,0] - xini# x(:,1) = x0       
        # equality constraints for dynamics
        for j in range(nx):
            eq_con[j,i+1] = x[:,i+1][j]-dice_dynamics(x[:,i],u[:,i],Params)[j]
    # extra constraints, leaving initial states for input shift as decision variables
    eq_con_extra = [xini[0:nx-3][i] - x0_ini[0:nx-3][i] for i in range(nx-3)]+[xini[-1] - x0_ini[-1]]
    
    eq_con_flat = []
    for i in range(N+1):
        for j in range(nx):
            eq_con_flat.append(eq_con[:,i][j])
    eq_con_ini = eq_con_flat+ eq_con_extra
    
    # define the objective (Mayer term)
    #obj = ((5 * 0.016408662 * Params.sc.J*x(nx, N+1)) - 3855.106895);
    obj = 1*((x[nx-1, N])) #--> improved scaling

    # define NLPs 
    
    x_flat = []
    for i in range(N+1):
        for j in range(nx):
            x_flat.append(x[:,i][j])
    u_flat=  []
    for i in range(N+1):
        for j in range(nu):
            u_flat.append(u[:,i][j])
    xini_flat = [xini[i] for i in range(nx)]
    
    # NLP for very first OCP
    nlp_ini={}
    nlp_ini['x']=x_flat+xini_flat+u_flat
    nlp_ini['f']=obj
    nlp_ini['g']=eq_con_ini


    # NLP for NMPC sequence of OCPs
    nlp={}
    nlp['x']=x_flat+u_flat
    nlp['f']=obj
    nlp['g']=eq_con_flat
    nlp['p']=xini_flat
    
    # ========================================================================
    # construct guess for states and inputs
    # =========================================================================
    u_guess = 0.5*np.ones((nu, N+1)).T
    x_guess = [x0]+[[0]*nx for i in range(N)]
    for i in range (N):
        x_guess[i+1] = dice_dynamics(x_guess[i],u_guess[i],Params)
    x_guess_flat = x_guess[0]
    for i in range(1,len(x_guess)):
        x_guess_flat = list(itertools.chain(x_guess_flat, x_guess[i])) 
    u_guess_flat = u_guess[0]
    for i in range(1,len(u_guess)):
        u_guess_flat = list(itertools.chain(u_guess_flat, u_guess[i])) 

    xu_guess = list(itertools.chain(x_guess_flat, u_guess_flat))
    xu_guess_ini = list(itertools.chain(x_guess_flat, x0))
    xu_guess_ini = list(itertools.chain(xu_guess_ini, u_guess_flat ))
    
    # ========================================================================
    # prepare output data
    # =========================================================================
    SingleOCP = {}
    SingleOCP['nlp'] = nlp_ini
    SingleOCP['xu_guess'] = xu_guess_ini
    SingleOCP['xu_LB'] = xu_LB_ini
    SingleOCP['xu_UB'] = xu_UB_ini
    SingleOCP['max_iter'] = 300
    
    SequenceOCP = {}
    SequenceOCP['nlp'] = nlp
    SequenceOCP['xu_guess'] = xu_guess
    SequenceOCP['xu_LB'] = xu_LB
    SequenceOCP['xu_UB'] = xu_UB
    SequenceOCP['max_iter'] = 300
    
    return SingleOCP, SequenceOCP

