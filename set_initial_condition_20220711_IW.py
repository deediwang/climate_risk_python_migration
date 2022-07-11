#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np

def set_initial_conditions(varargin):
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


# In[ ]:




