#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Preliminary: self_assign class; set_initial_conditions function; ConstructNLP function
#Haven't completely tested due to failure in casadi package and integration with preliminary functions not completed

import numpy as np
import time
from casadi import * 


# In[ ]:


#%% Define data of the DICE MPC-loop & construct NLP

# Set parameters using an appropriate function
#!External Function
Params = self_assign(versions='v2013')

Data={}
Data.N = Params.N  # Prediction horizon set in above script file
Data.step = 1      # 1 == 1-step MPC strategy
Data.t0 = 1        # initial simulation time; needs to be >=1
Data.tf = 20       # final simulation time; set tf = 1 to solve single OCP
Data.nx = 1+6+9+1  # total # of states: time; 6 "endogenous" states; 
                    #   9 auxilliary states including 5 "exogenous" states,
                    #   consumption, emissions, and shifted inputs;
                    #   objective
Data.nu = 2;       # # of inputs


#!External Function
Params.x0 = set_initial_conditions(Data.t0,Params) # get x0

print('============================================================')
print('                        MPC-DICE')
print('                   Copyright (C) 2018 by')
print('Timm Faulwasser, Christopher M. Kellett and Steven R. Weller')
print(' ')
print(' ')
print('============================================================')


# In[ ]:


#%% Symbolic problem construction
print('Symbolic construction of DICE optimal control problem ...')
tic = time.time()
#!External Function
SingleOCP, SequenceOCP = ConstructNLP(Data.N, Params.x0, Params) # construct NLPs
#!Casadi Internal Function
solver_ini = nlpsol('solver', 'ipopt', SingleOCP.nlp, Params.opts) # create IPOPT solver object    
solver     = nlpsol('solver', 'ipopt', SequenceOCP.nlp, Params.opts) # create IPOPT solver object
t_NLP = time.time() - tic

# Prepare output data struct
Data.state_list = ['time', 'T_AT', 'T_LO', 'M_AT', 'M_UP',                   'M_LO', 'K', 'sigma', 'L', 'A_TFP', 'Emission', 'C',                   'mu', 's', 'J']              
Data.xMPC  = []
Data.uMPC  = []
Data.wMPC  = []
Data.lam_C = []
Data.lam_E = []
Data.t_SOL = []


# In[ ]:


#%% Main loop

if Data.tf==1: #Solve single OCP
    #solve the NLP 
    tic = time.time()    
    #!Casadi Internal Function
    res = solver_ini( 'x0' , SingleOCP.xu_guess,\  # solution guess (hot start)
                        'lbx', SingleOCP.xu_LB,\   # lower bound on x
                        'ubx', SingleOCP.xu_UB,\   # upper bound on x
                        'lbg', 0, \     # lower bound on g
                        'ubg', 0)       # parameters of NLP 
    toc = time.time() - tic
    Data.t_SOL.append(toc)
    
    #?Not sure how solver_ini func return x in Python(sparse or full); Thus not sure whether need to use this 
    #xu_opt = full(res.x);
    xu_opt = res.x
    
    index_w = (Data.N+1)*Data.nu -1
    index_x = (Data.N+1)*Data.nx
    w_opt = xu_opt[-1-index_w:-1]        
    w_opt = np.reshape(w_opt, [Data.nu, (Data.N+1)])
    
    x_opt = xu_opt[:index_x]
    x_opt = np.reshape(x_opt, [Data.nx, (Data.N+1)])
    u_opt = x_opt[-1-2:-1-1, :]
    
    Data.xMPC = x_opt[:, :Data.N+1]
    Data.uMPC = u_opt
    Data.wMPC = w_opt
    
    #?Not sure how solver_ini func return lam_g in Python(sparse or full); Thus not sure whether need to use this
    #lambda = full(res.lam_g);
    lambda = res.Lam_g
    lambda = lambda(:-1-15)
        
    lambda = np.reshape(lambda, [Data.nx, Data.N+1])
    Data.lam_E = lambda(13,:) # dW/dE
    Data.lam_C = lambda(14,:)./Params.sc.C # dW/dC
    Data.SCC   = -1000*Data.lam_E./Data.lam_C
    
    # print computation times to screen
    print('============================================================')
    print('Time to construct NLP: %s    ', %(t_NLP))
    print('Time to solve single NLP: %s ', %(np.max(Data.t_SOL)))
    print('============================================================')
    
else: #Run MPC loop
    #!Haven't optimize this for loop
    for k in np.arange( Data.t0 , Data.tf, Data.step):
        print('MPC step k = %s of %s' , %(k,np.ceil(Data.tf/Data.step)))

        # assign initial condition
        if k > Data.t0:
            xk = Data.xMPC(:,-1) 

        # solve the NLP
        tic = time.time()
        if k == Data.t0: # very first OCP
            #!Casadi Internal Function
            res = solver_ini( 'x0' , SingleOCP.xu_guess,\# solution guess (hot start)
                                'lbx', SingleOCP.xu_LB,\ # lower bound on x
                                'ubx', SingleOCP.xu_UB,\ # upper bound on x
                                'lbg', 0,\               # lower bound on g
                                'ubg', 0)                # parameters of NLP                 
        else: # sequence OCPs
            #!Casadi Internal Function
            res = solver( 'x0' , xu_guess,\              # solution guess (hot start)
                            'lbx', SequenceOCP.xu_LB,\   # lower bound on x
                            'ubx', SequenceOCP.xu_UB,\   # upper bound on x
                            'lbg', 0,\                   # lower bound on g
                            'ubg', 0,\                   # upper bound on g
                            'p',   xk)                   # parameters of NLP                                                
        
        #?Again not sure need to use full()
        #xu_opt = full(res.x);
        xu_opt = res.x
        index_w = (Data.N+1)*Data.nu -1
        index_x = (Data.N+1)*Data.nx
        w_opt = xu_opt[-1-index_w:-1]       
        w_opt = np.reshape(w_opt, [Data.nu, (Data.N+1)])
        x_opt = xu_opt[:index_x];
        x_opt = np.reshape(x_opt, [Data.nx, (Data.N+1)])
            
        if k == Data.t0:
            Data.xMPC = x_opt[:,1]
                        
        u_opt = x_opt[-1-2:-1-1, np.arange(0,Data.step,1)] #actual input
        x_opt = x_opt[:, 1+np.arange(0,Data.step,1)] #state
        w_opt = w_opt[:, np.arange(0,Data.step,1)] # shifted input = optimization input
            
        tic=time.time()
        Data.t_SOL.append(time.time()-tic)
        Data.xMPC.append(x_opt)
        Data.uMPC.append(u_opt)
        Data.wMPC.append(w_opt)
            
        #?Again not sure need to use full()
        #lambda  = full(res.lam_g);
        lambda  = res.lam_g
        if k == Data.t0:
            lambda = lambda(1:-1-15)
            lambda = np.reshape(lambda, [Data.nx, Data.N+1])
        else:
            lambda = np.reshape(lambda, [Data.nx, Data.N+1])
            
        lambdaE = lambda(13,np.arange(0,Data.step,1)) #11
        lambdaC = lambda(14,np.arange(0,Data.step,1))/Params.sc.C #12
            
        Data.lam_C.append(lambdaC)
        Data.lam_E.append(lambdaE)

        # construct guess
        if k <= Data.tf:
            if k == 1:                    
                offset = (Data.N+1)*Data.nu + 1*Data.nx
                xu_guess = np.array([xu_opt[Data.nx+len(x_opt(:))+1:-1-offset],                                        x_opt[:],                                        xu_opt[-1-offset+len(u_opt[:])+1:-1],                                        u_opt[:]])
            else:
                offset = (Data.N+1)*Data.nu;
                xu_guess = np.array([xu_opt(len(x_opt[:])+1:-1-offset),                                        x_opt[:].                                        xu_opt(-1-offset+len(u_opt[:])+1:-1),                                        u_opt[:]])

    
# print computation times to screen
print('============================================================')
print('Time to construct NLP: $s     ', %(t_NLP))
print('Minimal time to solve NLP: %s ', %(np.min(Data.t_SOL)))
print('Maximal time to solve NLP: %s ', %(np.max(Data.t_SOL)))
print('Average time to solve NLP: %s ', %(np.mean(Data.t_SOL)))
print('Number of NLPs solved: %s     ', %(np.ceil(Data.tf/Data.step)))
print('============================================================')

