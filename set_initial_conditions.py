#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from casadi import *
import numpy as np
def set_initial_conditions(Params, t0, x_ini, have_ini):
    
    #This function is to set initial conditions for the model
    #Preliminary functions: assign_parameters
    #Input: params generated from parameters assignment, starting time, initial conditions, boolean: whether we have the third input
    #Output: x0 for Params (a single list)
    #Author: Ivan Wang 20220628, Jannik Jiang 20220713, Xiaoman Zhang 20220715
    
    if have_ini==False:
        # initial conditions for auxiliary state variables
        mu0 = Params.miu0; # initial mitigation rate
        s0= 0.259029014481802; #  initial savings rate
        
    else:  
        #initial conditions for auxiliary state variables
        mu0 = x_ini[14] # initial mitigation rate == casadi var
        s0= x_ini[15] # initial savings rate == casadi var

    x0 =np.transpose([Params.T_AT0, Params.T_LO0, Params.M_AT0, Params.M_UP0, Params.M_LO0, Params.K0])

    Gross_Economic_Output0 = Params.A0*(x0[5]**Params.gamma)*((Params.L0/1000)**(1-Params.gamma))
    Emissions0 = 5*(Params.sigma0*(1-mu0)*Gross_Economic_Output0 + Params.EL0)
    Damages0 = 1 / (1 + (Params.a2*(x0[0]**Params.a3)) + (Params.a4*x0[0])**Params.a5)
    theta10 = Params.sigma0 * (Params.pb/(1000*Params.theta2)) * (1-Params.deltaPB)**(t0-1)
    Net_Economic_Output0 = Damages0 * (1 - theta10*(mu0**Params.theta2))*Gross_Economic_Output0
    C0 = 5*(1-s0) * Net_Economic_Output0
    
    ## scaling         
    x0[3] = x0[3]/Params.sc['M']
    x0[2] = x0[2]/Params.sc['M']
    x0[4] = x0[4]/Params.sc['M']
    x0[5] = x0[5]/Params.sc['K']
    Params.L0 = Params.L0/Params.sc['L']
    C0 = C0/Params.sc['C']
    x_zero=[t0]+list(x0)+[Params.sigma0]+[Params.L0]+[Params.A0]+[Params.EL0]+[Params.f0]+[Emissions0]+[C0[0]]+[mu0]+[s0]+[0]
    
    return x_zero

