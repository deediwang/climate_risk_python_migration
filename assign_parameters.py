#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy.random import seed
from numpy.random import normal
from numpy.random import gamma
seed(1)

class assign_parameters():
    def __init__(self,versions):
        
        #This code is to initialize the parameters values for DICE model
        #Input: the version of the MATLAB file
        #Output: the initilal values for the parameters saved in the object
        #Author: Ivan Wang 20220630, Xiaoman Zhang 20220701
        
        # Constants
        self.N = 60                 # Horizon length
        self.BaseYear = 2010        # First calendar year
        self.eta = 3.8              # Forcings of equilibrium CO2 doubling (GAMS fco22x)
        self.M_AT_Base = 588        # Base atm carbon concentration (GAMS in FORC(t) eqn)
        self.deltaK = 0.1           # Capital depreciation (5 year) (GAMS dk)
        self.gamma = 0.3            # Capital elasticity in production function (GAMS gama)
        self.theta2 = 2.8           # Exponent of control cost function (GAMS expcost2)
        self.a3 = 2                 # Damage exponent
        self.alpha = 1.45           # Elasticity of marginal utility of consumption (GAMS elasmu)
        self.rho = 0.015            # Initial rate of social time preference per year (GAMS prstp)
        self.xi1 = 0.098            # Climate equation coefficient for upper level (GAMS c1)
        self.xi2 = 3/11             # Conversion factor from GtC to CtCO2
        self.limmiu = 1.2           # Upper limit on emissions drawdown
        
        # Climate Model Diffusion Parameters
        c3 = 0.088
        c4 = 0.025
        
        # Carbon Cycle Model Diffusion Parameters
        b12 = 0.088
        b23 = 0.0025
        mateq = 588
        mueq = 1350
        mleq = 10000
        zeta11 = 1 - b12
        zeta21 = b12
        zeta12 = (mateq/mueq)*zeta21
        zeta22 = 1 - zeta12 - b23
        zeta32 = b23
        zeta23 = zeta32*(mueq/mleq)
        zeta33 = 1 - zeta23
        # carbon concentration matrix calculation
        self.Phi_M = [[zeta11,zeta12,0],[zeta21,zeta22,zeta23], [0,zeta32,zeta33]]
        
        # Exogenous Signal Constants
        self.L0 = 6838           # Initial population
        self.La = 10500          # Asymptotic population
        self.lg = 0.134          # Population growth rate
        self.EL0 = 3.3           # Initial land use emissions
        self.deltaEL = 0.2       # Land use emissions decrease rate
        self.A0 = 3.8            # Initial Total Factor Productivity (TFP)
        self.deltaA = 0.006      # TFP increase rate
        self.deltaPB = 0.025     # Decline rate of backstop price
        e0 = 33.61               # Initial emissions
        q0 = 63.69               # Initial global output
        self.miu0 = 0.039        # Initial mitigation rate
        self.sigma0 = e0/(q0*(1-self.miu0)) # Calculated initial emissions intensity
        self.gsigma = 0.01       # Emissions intensity base rate
        self.deltasigma = 0.001  # Decline rate of emissions intensity
        self.f0 = 0.25           # Initial forcings of non-CO2 GHGs
        self.f1 = 0.7            # Forcings of non-CO2 GHGs in 2100
        self.tforce = 18         # Slope of non-CO2 GHG forcings
        self.optlrsav = (self.deltaK + 0.004)/(self.deltaK+ 0.004*self.alpha + self.rho) * self.gamma
        
        # Initial condition for states
        self.T_AT0 = 0.8
        self.T_LO0 = 0.0068
        self.M_AT0 = 830.4
        self.M_UP0 = 1527
        self.M_LO0 = 10010
        self.K0 = 135
        
        # scaling of selected state variables required for numerical stability
        sc={}
        sc['L'] = 100
        sc['M'] = 100
        sc['J'] = 10000
        sc['K'] = 100
        sc['C'] = 100
        self.sc = sc   

            
        if versions=='v2013' or versions=='template':
            
            if versions=='v2013':
                # the saved file from the model run
                self.parameter_set = 'v2013'
                # Limit on maximum atmospheric temperature
                self.T_AT_max = 2
                
            elif versions=='template':
                # the saved file from the model run
                self.parameter_set = 'template'
                # Limit on maximum atmospheric temperature
                self.T_AT_max = 20

            # Constants
            self.a2 = 0.00267     # Damage multiplier
            
            # Climate Model Diffusion Parameters
            self.t2xco2 = 2.9
            phi11 = 1-self.xi1*((self.eta/self.t2xco2) + c3)
            phi12 = self.xi1*c3
            phi21 = c4
            phi22 = 1-c4
            # temprature anomalies matrix calculation
            self.Phi_T = [[phi11,phi12], [phi21, phi22]]
            
            # Exogenous Signal Constants
            self.ga = 0.079          # Initial TFP rate
            self.pb = 344            # Initial backstop price
            
            # Options for IPOPT
            opts={}
            opts['ipopt'] = {}
            opts['ipopt']['max_iter'] = 3000
            opts['ipopt']['print_level'] = 5 #0,3
            opts['print_time'] = 0
            opts['ipopt']['acceptable_tol'] = 1e-8
            opts['ipopt']['acceptable_obj_change_tol'] = 1e-8
            self.opts = opts
            
        elif versions=='cf':
            # the saved file from the model run
            self.parameter_set = 'cf'
            
            # Limit on maximum atmospheric temperature
            self.T_AT_max = 60
            
            # Constants
            self.a2 = 0           # Damage multiplier
            self.a4 = normal(loc=0.12, scale=0.04, size=1) # Stochastic damage function
            self.a5 = 7 # Stochastic damage function
            
            # Climate Model Diffusion Parameters
            self.t2xco2 = 1/gamma(shape = 1.54, scale=1/0.9, size=1)+0.75
            phi11 = 1-self.xi1*((self.eta/self.t2xco2) + c3)
            phi12 = self.xi1*c3
            phi21 = c4
            phi22 = 1-c4
            # temprature anomalies matrix calculation
            self.Phi_T = [[phi11,phi12], [phi21, phi22]]
            
            # Exogenous Signal Constants
            self.ga = normal(loc=0.0084*5, scale=0.0059*np.sqrt(5), size=1)  # Initial TFP rate
            self.pb = normal(loc=343, scale=137, size = 1);           # Initial backstop price
            
            # Options for IPOPT
            opts={}
            opts['ipopt'] = {}
            opts['ipopt']['max_iter'] = 300
            opts['ipopt']['print_level'] = 0 #0,3
            opts['print_time'] = 0
            opts['ipopt']['acceptable_tol'] = 1e-8
            opts['ipopt']['acceptable_obj_change_tol'] = 1e-8
            opts['ipopt']['max_cpu_time'] = 3
            self.opts = opts
        

params_2013 = assign_parameters('v2013')
params_cf = assign_parameters('cf')


