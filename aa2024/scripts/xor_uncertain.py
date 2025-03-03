from gurobipy import Model, GRB, quicksum

def AddBias(X):
    return [x for x in X] + [1]

def Sign(x):
    return 1 if x >= 0.01 else -1

from numpy.random import normal, seed
seed(13)
def AddNoise(X, mu=0.1):
    return list(map(lambda x: x+normal(0, mu), X))


class XorBatch(object):
    def __init__(self, NH=2) -> None:
        self.NH = NH
        self.mu = 0

    
    def solve(self, Xs, Ys, Phi=lambda x: x, timelimit=10):
        # Map input to feature space
        Xs = [Phi(x) for x in Xs]

        model = Model()
        model.setParam(GRB.Param.NonConvex, 2)
        model.setParam(GRB.Param.OutputFlag, 1)
        model.setParam(GRB.Param.TimeLimit, 5)
        
        # Number of samples
        n = len(Xs)
        # Input dimension
        m = len(Xs[0])
        
        nh = self.NH

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
        z = {}
        for k in range(n):
            for h in range(nh):
                z[k, h] = model.addVar(vtype=GRB.BINARY)
        
        M = 1000

        # First layer: input x_ki to hidden units z_kh
        for k in range(n):
            for h in range(nh):
                model.addConstr( quicksum(Xs[k][i]*(wp[i,h] - wn[i,h]) for i in range(m)) >= 0.01 - M*(1 - z[k,h]) )
                model.addConstr( quicksum(Xs[k][i]*(wp[i,h] - wn[i,h]) for i in range(m)) <= 0.00 + M*z[k,h] )

        # linearization constraints
        for k in range(n):
            model.addConstr( quicksum((2*z[k,h] - 1)*(up[h] - un[h]) for h in range(nh)) >= 0.01 - M*(1-y_hat[k]))
            model.addConstr( quicksum((2*z[k,h] - 1)*(up[h] - un[h]) for h in range(nh)) <= 0.00 + M*y_hat[k])
                
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

        if self.mu > 0:
            for i, h in wp:
                wp[i, h].VarHintVal = self.wp_bar[i, h]/self.mu
                wn[i, h].VarHintVal = self.wn_bar[i, h]/self.mu 

            for i in range(len(up)):
                up[i].VarHintVal = self.up_bar[i]/self.mu
                un[i].VarHintVal = self.un_bar[i]/self.mu
                
        model.optimize()
        
        if model.status != GRB.OPTIMAL and model.status != GRB.TIME_LIMIT and model.status != GRB.USER_OBJ_LIMIT:
            return None
        
        if self.mu == 0:
            self.mu = 1
            self.wp_bar = {(i,h): wp[i,h].x for i,h in wp}
            self.wn_bar = {(i,h): wn[i,h].x for i,h in wn}
            self.up_bar = [up[h].x for h in range(nh)]
            self.un_bar = [un[h].x for h in range(nh)]
        else:
            self.mu += 1
            self.wp_bar = {(i,h): (wp[i,h].x + self.wp_bar[i,h]) for i,h in wp}
            self.wn_bar = {(i,h): (wn[i,h].x + self.wn_bar[i,h]) for i,h in wn}
            self.up_bar = [(up[h].x + self.up_bar[h]) for h in range(nh)]
            self.un_bar = [(un[h].x + self.un_bar[h]) for h in range(nh)]

        wbar = {(i, h): wp[i, h].X - wn[i,h].X for i, h in wp}
        ubar = [up[h].X - un[h].X for h in range(nh)]

        def F(X):
            X = Phi(X)
            return Sign(sum(ubar[h] * Sign(sum(X[i]*wbar[i, h] for i in range(m))) for h in range(nh)))
        
        return F

if __name__ == "__main__":
    Xxor = [(-1, -1), (-1, 1), (1, -1), (1, +1)]
    Yxor = [-1, +1, +1, -1]
 
    Xtrain, Ytrain = [], []
    for _ in range(20):
        Xtrain += [AddNoise(x) for x in Xxor]
        Ytrain += Yxor

    solver = XorBatch(NH=4)
    Predict = solver.solve(Xtrain, Ytrain)

    acc = 0
    for x,y in zip(Xtrain, Ytrain):
        if y == Predict(x):
            acc += 1
    print('Train accuracy:', acc/len(Xtrain))

    Xtest, Ytest = [], []
    for _ in range(20):
        Xtest += [AddNoise(x) for x in Xxor]
        Ytest += Yxor
    acc = 0
    for x,y in zip(Xtest, Ytest):
        if y == Predict(x):
            acc += 1
    print('Test accuracy:', acc/len(Xtest))


