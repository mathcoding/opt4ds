import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import *


def MIP(Xs,Ys,nh):
    
    model = ConcreteModel()
    
    m=len(Xs[0]) # numero input che abbiamo
    model.I=RangeSet(0,m-1)
    n=len(Xs)
    model.K=RangeSet(0,n-1)
    
    model.H=RangeSet(0,nh-1) # parametri nascosti
    
    
    # variabili
    model.W=Var(model.I, model.H, within=Reals) 
    model.U=Var(model.H, within=Reals)
    
    model.s=Var(model.K, model.H, within=Binary)
    model.z=Var(model.K, model.H, within=Reals)
    
    model.y_hat=Var(model.K, within=Binary)
    model.alpha=Var(model.K, within=NonNegativeReals)
    
    model.obj=Objective(expr=sum(0.5*model.alpha[k] for k in model.K))
    
    #vincoli perceptron
    M = 1000
                          
    model.Margin = ConstraintList()
    for j in model.H:
        for k in model.K:
            model.Margin.add( sum(Xs[k][i]*model.W[i,j] for i in model.I) >= 0.0 - M*(1 - model.s[k,j]) )
            model.Margin.add( sum(Xs[k][i]*model.W[i,j] for i in model.I) <= -0.1 - M*model.s[k,j]) 
            
    # linearization constraints
    model.Linearize=ConstraintList()
    for k in model.K:
        for j in model.H:
            model.Linearize.add( model.z[k,j] <= model.U[j] + M*(1-model.s[k,j]))
            model.Linearize.add( model.z[k,j] >= model.U[j] - M*(1-model.s[k,j]))
            model.Linearize.add( model.z[k,j] <= M*model.s[k,j])
            model.Linearize.add( model.z[k,j] >= -M*model.s[k,j])
            
    model.Neurons = ConstraintList()
    for k in model.K:
        model.Neurons.add( sum((-model.U[j] + 2*model.z[k,j]) for j in model.H) >= 0.0 -M*(1-model.y_hat[k]))
        model.Neurons.add( sum((-model.U[j] + 2*model.z[k,j]) for j in model.H) <= -0.1 +M*model.y_hat[k])
            
   # norm constraint
    model.norm1= ConstraintList()
    for k in model.K:
       model.norm1.add( (2*model.y_hat[k]-1) -Ys[k] <= model.alpha[k])
       model.norm1.add( Ys[k] - (2*model.y_hat[k]-1) <= model.alpha[k])
       
    sol = SolverFactory('gurobi').solve(model, tee=True)
    
    sol_json = sol.json_repn()
    
    if sol_json['Solver'][0]['Status'] != 'ok':
        return None, None
    if sol_json['Solver'][0]['Termination condition'] != 'optimal':
        return None, None
     
    W=np.ones((m,nh))
    U=np.ones(nh)
    for i in model.I:
        for j in model.H:
            W[i,j]=model.W[i,j]()
    for j in model.H:
        U[j]=model.U[j]()
        
    return W,U
       
def Rho(x):
    if x >= 0:
        return 1
    return -1


if __name__ == "__main__":
    # from np.random import seed
    # seed(13)
    
    Xxor=[(1,-1,-1),(1,-1,1),(1,+1,-1),(1,+1,+1)]
    Yxor=[1,-1,-1,+1]
 
    nh = 2
    W,U=MIP(Xxor,Yxor,2)

    print('W',W)
    print('U',U)
    # to verify the model
    for x,y in zip(Xxor,Yxor):
        print(y, Rho(sum(U[j]*Rho(sum(x[i]*W[i,j] for i in range(3))) for j in range(nh))))
            