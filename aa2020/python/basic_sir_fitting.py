# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 09:23:13 2020

@author: Gualandi

Example originally adapted from Pyomo, Chap 7.4.3, pag. 135.

"""


# Import the libraries
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory
from pyomo.environ import maximize, minimize, Binary, RangeSet, ConstraintList
from pylab import figure, show, ylabel, xlabel, savefig, title, ylim


def PlotRuntimeNodes(Rs, L2, L1, Name):
    """ Plots real cases and fitted models """
    Xs = [i+1 for i in range(len(C))]
    X2 = [i+1 for i in range(len(L2))] 
    X1 = [i+1 for i in range(len(L1))] 
    
    # create the general figure
    fig1 = figure()
    ax1 = fig1.add_subplot(111)
    ylim((0,45000))
    ylabel("Infecteds")
    xlabel("Time Units")
    title(Name.upper())
     
    line3 = ax1.plot(X1, L1, 'g-', alpha=0.5, label='SIR - $L_1$ loss')    
    line2 = ax1.plot(X2, L2, 'b-', alpha=0.5, label='SIR - $L_2$ Loss')    
    line1 = ax1.plot(Xs, Rs, 'or-', alpha=0.5, 
                     label='Real Cases')
    ls = line1 + line2 + line3
    
    labs = [l.get_label() for l in ls]
    ax1.legend(ls, labs, loc=0)

    ax1.axhline(y=1000, color='grey', linestyle='--', alpha=0.5)
    ax1.axvline(x=60, color='grey', linestyle='--', alpha=0.5)
    
    savefig(Name+".pdf", bbox_inches='tight')
    
    show()
    
    
def F(x):
    return int(sum([max(0, x[i+1]-x[i]) for i in range(len(x)-1)]))

def MovingAvg(Ls, w=3):
    """ Moving average of input data, with time window of w days """
    Ws = [0 for i in range(w-1)] + Ls
    return [sum(Ws[i:i+w])/w for i in range(len(Ws)-w+1)]

    
def DecomposeWeek(C, n=7):
    """ Change time unit of periods in n days """
    d = len(C) // n + 1
    idx = 0
    W = []
    for i in range(d):
        w = 0
        for _ in range(n):
            w += C[min(idx, len(C)-1)]
            idx += 1
        W.append(w)
    return W
    

def FittingSIR(N, C, H, DomBeta, DomGamma, R0, Loss=2):
    """ Input parameters for SIR model:
        
           N: population size
           C: observed infected cases
           H: time Horizon
           DomBeta: domain for beta
           DomGamma: domain for gamma
           Loss: type of loss function for fitting 
    """
           
    # Main Pyomo model
    model = ConcreteModel()
    # Time Horizon for data fitting
    F = len(C)
    # Parameters
    model.T = RangeSet(H)
    model.T1 = RangeSet(2, H)
    model.T2 = RangeSet(F)
    # Variables of SIR model
    model.I = Var(model.T, bounds=(0, N), initialize=1) 
    model.S = Var(model.T, bounds=(0, N), initialize=N) 
    # Variables: Parameters to be estimated
    model.beta  = Var(bounds=DomBeta,  initialize=1.0) 
    model.gamma = Var(bounds=DomGamma, initialize=1.0) 
    # Variables for fitting
    model.eps = Var(model.T2, initialize=0.0)
    
    # Constraint on the susceptile dynamics
    def _Susceptile(model, i): 
        return model.S[i] == (1 - model.beta/N*model.I[i-1])*model.S[i-1] 
    model.Susceptile = Constraint(model.T1, rule = _Susceptile)

    # Constraint on the infected dynamics
    def _Infected(model, i): 
        return model.I[i] == (1 - model.gamma + model.beta/N*model.S[i-1])*model.I[i-1]    
    model.Infected = Constraint(model.T1, rule = _Infected)

    # Constraint on R0    
    model.Ratio = Constraint(expr = model.beta <= R0*model.gamma)

    # Diferente objective and constraint depending the Loss function
    if Loss == 2: # Loss Norm 2: Least Square
        model.objective = Objective(expr=sum((model.eps[i])**2 for i in model.T2))
        
        # Fitting the data given in input
        def _Data(model, i): 
            return C[i-1] == model.I[i] + model.eps[i] 
        model.Data = Constraint(model.T2, rule  = _Data )
        
    else: # Loss Norm 1: sum of ABS
        model.objective = Objective(expr=sum(model.eps[i] for i in model.T2))

        # Fitting the data given in input
        def _Data1(model, i): 
            return + C[i-1] - model.I[i] <= model.eps[i]     
        model.Data1 = Constraint(model.T2, rule = _Data1)
     
        def _Data2(model, i): 
            return - C[i-1] + model.I[i] <= model.eps[i]     
        model.Data2 = Constraint(model.T2, rule = _Data2)
        
    
    SolverFactory('ipopt').solve(model)
    
    print('beta: {:.2f}, gamma: {:.2f}, R0: {:.2f}'
          .format(model.beta(), model.gamma(), model.beta()/model.gamma()))
    
    
    print(model.I[1](), C[0])
    
    return model.gamma(), model.beta(), [model.I[i]() for i in model.T]


# -----------------------------------------------
#   MAIN function
# -----------------------------------------------
if __name__ == "__main__":
    # Lombardi+Veneto+Emilia Romagna (31/03/2020)    
    TriRegion = [1, 5, 20, 62, 155, 216, 299, 364, 554, 766, 954, 1425, 1672,
            2021, 2358, 2815, 3278, 4184, 5092, 6470, 6627, 8291, 9951,
            11196, 13183, 14773, 16223, 17987, 19134, 21613, 24186, 27245,
            28919, 31116, 32930, 34592, 37179,
            39904, 41386, 43178, 43336, 43927]

    # Bound for paramaters
    DomBeta = (0.1, 100)
    DomGamma = (0.1, 10)
    R0 = 4.0
    
    # Problem parameters
    smooth = 5
    timeunit = 1
    Days = 90
    Horizon = 90//timeunit
    
    C = MovingAvg(TriRegion, smooth)
    # C = DecomposeWeek(C, timeunit)
   
    # Population Size
    N = 20*10**6 # For China
    
    # Fit Model
    gamma2, beta2, L2 = FittingSIR(N, C, Horizon, DomBeta, DomGamma, R0, 2)
    gamma1, beta1, L1 = FittingSIR(N, C, Horizon, DomBeta, DomGamma, R0, 1)
    
    # Plot findings
    PlotRuntimeNodes(C, L2, L1, "china")
    print('Infetti:', F(C), ' Errore:', F(L2)-F(C), F(L1)-F(C))
