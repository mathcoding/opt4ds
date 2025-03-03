from gurobipy import Model, GRB, quicksum

def AddBias(X):
    return [x for x in X] + [1]

def Sign(x):
    return 1 if x >= 0.01 else -1

from numpy.random import normal
def AddNoise(X, mu=0.1):
    return list(map(lambda x: x+normal(0, mu), X))

def MIPxor_Linearized(Xs, Ys, Phi=lambda x: x, NH=2):
    # Map input to feature space
    Xs = [Phi(x) for x in Xs]

    model = Model()
    
    # Number of samples
    n = len(Xs)
    # Input dimension
    m = len(Xs[0])

    # Hidden parameters
    nh = NH
        
    # Variables: w_ih weights of fisrt layer
    w = {}
    for i in range(m):
        for h in range(nh):
            w[i, h] = model.addVar(vtype=GRB.BINARY)
    # Variables: u_h weights of second layer
    u = [model.addVar(vtype=GRB.BINARY) for j in range(nh)]

    y_hat = [model.addVar(vtype=GRB.BINARY) for k in range(n)]
    alpha = [model.addVar(lb=0, obj=1) for k in range(n)] 
    
    # Auxiliary variables for mapping activation function sign(x^T w)
    z = {}
    v = {}
    for k in range(n):
        for h in range(nh):
            z[k, h] = model.addVar(vtype=GRB.BINARY)
            v[k, h] = model.addVar(vtype=GRB.BINARY)
    
    M = 1000

    # First layer: input x_ki to hidden units z_kh
    for k in range(n):
        for h in range(nh):
            model.addConstr( quicksum(Xs[k][i]*(2*w[i,h]-1) for i in range(m)) >= 0.01 - M*(1 - z[k,h]) )
            model.addConstr( quicksum(Xs[k][i]*(2*w[i,h]-1) for i in range(m)) <= 0.00 + M*z[k,h] )

    # linearization constraints
    for k in range(n):
        model.addConstr( quicksum((4*v[k,h] - 2*z[k,h] - 2*u[h] + 1) for h in range(nh)) >= 0.01 - M*(1-y_hat[k]))
        model.addConstr( quicksum((4*v[k,h] - 2*z[k,h] - 2*u[h] + 1) for h in range(nh)) <= 0.00 + M*y_hat[k])    

    for k in range(n):
        for h in range(nh):
            model.addConstr( v[k,h] >= z[k,h] + u[h] - 1 )
            model.addConstr( v[k,h] <= z[k,h] )
            model.addConstr( v[k,h] <= u[h] )

    for k in range(n):
        model.addConstr( (2*y_hat[k]-1) - Ys[k] <= alpha[k])
        model.addConstr( Ys[k] - (2*y_hat[k]-1) <= alpha[k])
    
    model.optimize()
     
    if model.status != GRB.OPTIMAL:
        return None
    
    wbar = {(i, h): 2*w[i, h].X - 1 for i, h in w}
    ubar = [2*u[h].X - 1 for h in range(nh)]

    print('wbar', sum(wbar[i,h] != 0 for i, h in wbar), wbar)
    print('ubar', sum(u != 0 for u in ubar), ubar)

    def F(X):
        X = Phi(X)
        return Sign(sum(ubar[h] * Sign(sum(X[i]*wbar[i, h] for i in range(m))) for h in range(nh)))
    
    return F

def MIPxor_NSA(Xs, Ys, Phi=lambda x: x, NH=2):
    # Map input to feature space
    Xs = [Phi(x) for x in Xs]

    model = Model()
    
    model.setParam(GRB.Param.NonConvex, 2)

    # Number of samples
    n = len(Xs)
    # Input dimension
    m = len(Xs[0])

    # Hidden parameters
    nh = NH
        
    # Variables: w_ih weights of fisrt layer
    wp, wn = {}, {}
    for i in range(m):
        for h in range(nh):
            wp[i, h] = model.addVar(obj=0.01, vtype=GRB.BINARY)
            wn[i, h] = model.addVar(obj=0.01, vtype=GRB.BINARY)
    # Variables: u_h weights of second layer
    up = [model.addVar(obj=0.01, vtype=GRB.BINARY) for j in range(nh)]
    un = [model.addVar(obj=0.01, vtype=GRB.BINARY) for j in range(nh)]

    y_hat = [model.addVar(vtype=GRB.BINARY) for k in range(n)]
    alpha = [model.addVar(lb=0, obj=1) for k in range(n)] 
    
    # Auxiliary variables for mapping activation function sign(x^T w)
    zp, zn = {}, {}
    for k in range(n):
        for h in range(nh):
            zp[k, h] = model.addVar(obj=0.01, vtype=GRB.BINARY)
            #zn[k, h] = model.addVar(obj=0.01, vtype=GRB.BINARY)
    
    M = 1000

    # First layer: input x_ki to hidden units z_kh
    for k in range(n):
        for h in range(nh):
            model.addConstr( quicksum(Xs[k][i]*(wp[i,h] - wn[i,h]) for i in range(m)) >= 0.01 - M*(1 - zp[k,h]) )
            model.addConstr( quicksum(Xs[k][i]*(wp[i,h] - wn[i,h]) for i in range(m)) <= 0.00 + M*zp[k,h] )

    # linearization constraints
    for k in range(n):
        model.addConstr( quicksum((2*zp[k,h] - 1)*(up[h] - un[h]) for h in range(nh)) >= 0.01 - M*(1-y_hat[k]))
        model.addConstr( quicksum((2*zp[k,h] - 1)*(up[h] - un[h]) for h in range(nh)) <= 0.00 + M*y_hat[k])
            
    for h in range(nh):
        model.addConstr( up[h] + un[h] <= 1)

    for i, h in wp:
        model.addConstr( wp[i,h] + wn[i,h] <= 1)

    for k in range(n):
        model.addConstr( (2*y_hat[k]-1) - Ys[k] <= alpha[k])
        model.addConstr( Ys[k] - (2*y_hat[k]-1) <= alpha[k])
    
    # wbar 4 {(0, 0): -1.0, (0, 1): 1.0, (1, 0): 1.0, (1, 1): -1.0}
    #model.addConstr( wn[0,0] == 1 )
    #model.addConstr( wp[0,1] == 1 )
    #model.addConstr( wp[1,0] == 1 )
    #model.addConstr( wn[1,1] == 1 )

    # ubar 2 [-1.0, -1.0]
    #model.addConstr( un[0] == 1 )
    #model.addConstr( un[1] == 1 )

    model.optimize()
     
    if model.status != GRB.OPTIMAL:
        return None
    
    wbar = {(i, h): wp[i, h].X - wn[i,h].X for i, h in wp}
    ubar = [up[h].X - un[h].X for h in range(nh)]

    print('wbar', sum(wbar[i,h] != 0 for i, h in wbar), wbar)
    print('ubar', sum(u != 0 for u in ubar), ubar)

    def F(X):
        X = Phi(X)
        return Sign(sum(ubar[h] * Sign(sum(X[i]*wbar[i, h] for i in range(m))) for h in range(nh)))
    
    return F


if __name__ == "__main__":
    Xxor = [(-1, -1), (-1, 1), (1, -1), (1, +1)]
    Yxor = [-1, +1, +1, -1]
 
    Xtrain = [AddNoise(x) for x in Xxor]
    F = MIPxor_NSA(Xtrain, Yxor, NH=4, Phi=AddBias)
    #F = MIPxor_Linearized(Xxor, Yxor, NH=2)

    # Provare togliendo il commmento per aggiungere un bias
    # F = MIPxor(Xxor, Yxor, AddBias)

    # to verify the model
     
    for x,y in zip(Xtrain, Yxor):
        print(x, y, F(x))
    print()
    for x,y in zip(Xxor, Yxor):
        print(x, y, F(x))
            