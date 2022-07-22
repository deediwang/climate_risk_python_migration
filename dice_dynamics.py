#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def dice_dynamics(x,u,Params):

    sc = Params.sc # scaling parameters for states

    i    = x[0] # time index considered as state variable
    T_AT = x[1] 
    T_LO = x[2]
    M_AT = sc['M']*x[3]
    M_UP = sc['M']*x[4]
    M_LO = sc['M']*x[5]
    K    = sc['K']*x[6]

    # Time-varying parameters reformulated as additional states
    sigma = x[7]
    L     = sc['L']*x[8]
    A_TFP = x[9]
    E_LAND = x[10]
    F_EX   = x[11]
    E     = x[12]
    C     = sc['C']*x[13]

    # Controls reformulated as states
    mu = x[14]
    s  = x[15]

    # Objective
    J = sc['J']*x[16] # scale obj 

    T = [T_AT, T_LO]
    M = [M_AT, M_UP, M_LO]

    # ------------------------------------------------------------------------
    #                              Controls
    # -------------------------------------------------------------------------

    # Shifted Mitigation Rate
    mu_NEXT = u[0]

    # Shifted Savings Rate
    s_NEXT = u[1]

    # ------------------------------------------------------------------------
    #                           Unpack Parameters
    # -------------------------------------------------------------------------
    eta         = Params.eta
    M_AT_Base   = Params.M_AT_Base
    deltaK      = Params.deltaK
    gamma       = Params.gamma
    theta2      = Params.theta2
    alpha       = Params.alpha
    rho         = Params.rho
    xi1         = Params.xi1
    xi2         = Params.xi2
    Phi_T       = Params.Phi_T
    Phi_M       = Params.Phi_M
    zeta11      = Phi_M[0][0]
    zeta12      = Phi_M[0][1]
    a2          = Params.a2
    a3          = Params.a3
    a4          = Params.a4
    a5          = Params.a5

    La          = Params.La          # Asymptotic population
    lg          = Params.lg          # Population growth rate

    EL0         = Params.EL0         # Initial land use emissions
    deltaEL     = Params.deltaEL     # Land use emissions decrease rate

    ga          = Params.ga          # Initial TFP rate
    deltaA      = Params.deltaA      # TFP increase rate

    pb          = Params.pb          # Initial backstop price
    deltaPB     = Params.deltaPB     # Decline rate of backstop price

    gsigma      = Params.gsigma      # Emissions intensity base rate
    deltasigma  = Params.deltasigma  # Decline rate of emissions intensity

    f0          = Params.f0          # Initial forcings of non-CO2 GHGs
    f1          = Params.f1          # Forcings of non-CO2 GHGs in 2100
    tforce      = Params.tforce      # Slope of non-CO2 GHG forcings

    # Dynamics / state recursion
    i_NEXT = i+1 #time index
    F_EX_NEXT = f0 + fmin(f1-f0, (f1-f0)*(i)/tforce)

    # Named functional quantities
    theta1 = sigma * (pb/(1000*theta2) * (1-deltaPB)**(i-1))
    Gross_Economic_Output = A_TFP*(K**gamma)*((L/1000)**(1-gamma))
    Damages = 1 / (1 + (a2*(T_AT**a3)) + (a4*T_AT)**a5)
    Net_Economic_Output = Damages *(1 - theta1*(mu**theta2))*Gross_Economic_Output

    # ---------------- Climate ------------------------
    Radiative_Forcing = xi1*(eta*np.log((zeta11*M_AT + zeta12*M_UP + xi2*E)/M_AT_Base)/np.log(2) + F_EX_NEXT)
    T_NEXT = SX_multiply(Phi_T, T)
    T_NEXT[0]+=Radiative_Forcing

    # ---------------- Carbon Cycle -------------------
    M_NEXT = SX_multiply(Phi_M, M)
    M_NEXT[0]+=xi2*E
    # ---------------- Economy ------------------------
    K_NEXT = (1 - deltaK)**5 * K + 5 * Net_Economic_Output * s
    # ---------------- Auxilliary States --------------
    # Time-varying parameters
    sigma_NEXT = sigma * np.exp(-gsigma * (((1-deltasigma)**5)**(i-1)) * 5)
    L_NEXT = L * (La/L)**lg
    A_TFP_NEXT = A_TFP / (1 - ga * np.exp(-deltaA * 5 * (i-1)))
    E_LAND_NEXT = EL0*(1-deltaEL)**(i)

    # Intermediate variables
    Gross_Economic_Output_NEXT = A_TFP_NEXT*(K_NEXT**gamma)*((L_NEXT/1000)**(1-gamma))
        
    T_AT_NEXT = T_NEXT[0]
    Damages_NEXT = 1 / (1 + (a2*(T_AT_NEXT**a3)) + (a4*T_AT_NEXT)**a5) 
    theta1_NEXT = sigma_NEXT * (pb/(1000*theta2)) * (1-deltaPB)**(i)
    Net_Economic_Output_NEXT = Damages_NEXT *(1 - theta1_NEXT*(mu_NEXT**theta2))*Gross_Economic_Output_NEXT
    
    # Emission and Consumption (needed for SCC computation)
    E_NEXT = 5*(sigma_NEXT*(1-mu_NEXT)*Gross_Economic_Output_NEXT + E_LAND_NEXT)
        
    C_NEXT = 5*(1-s_NEXT) * Net_Economic_Output_NEXT

    # ---------------- Objective ----------------------
    J_NEXT = J - L*(((1000/L*C)**(1-alpha) - 1)/((1-alpha)))/(1+rho)**(5*(i-1))        

    # numerical scaling
    M_NEXT = [m/sc['M'] for m in M_NEXT]
    L_NEXT = L_NEXT/sc['L']
    K_NEXT = K_NEXT/sc['K']
    C_NEXT = C_NEXT/sc['C']
    J_NEXT = J_NEXT/sc['J']

    f = [i_NEXT]+T_NEXT+ M_NEXT+[ K_NEXT, sigma_NEXT, L_NEXT, A_TFP_NEXT, E_LAND_NEXT, F_EX_NEXT, E_NEXT, C_NEXT, mu_NEXT, s_NEXT, J_NEXT]
    return f

