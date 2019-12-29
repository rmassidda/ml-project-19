from concurrent.futures import ThreadPoolExecutor
from grid import Grid
from network import Network
from utils import onehot
import numpy as np
import sys

class Validation:
    def __init__(self,train_x,train_y,loss,verbose=True):
        self.train_x = train_x
        self.train_y = train_y
        self.loss = loss
        self.verbose = verbose

    def hold_out(self,par):
        nn = Network(**par)

        # Split with validation set (75/25)
        bound = int(len(self.train_x)*(3/4))

        nn.train(self.train_x[:bound],self.train_y[:bound])

        valid_x = self.train_x[bound:]
        valid_y = self.train_y[bound:]

        estimated_risk = nn.error(valid_x,valid_y,self.loss)
        return par,estimated_risk

    def model_selection(self,hp):
        grid = Grid(hp)
        risk = np.Inf
        best = None

        # Parallel grid search
        if self.verbose:
            print("=== MODEL SELECTION ===")
        with ThreadPoolExecutor(max_workers=8) as executor:
            grid_res = executor.map(self.hold_out,grid)
            for p in grid_res:
                if self.verbose:
                    print(p,sep='\t')
                if p[1] < risk:
                    risk = p[1]
                    best = p[0]
        return best

    def model_assessment(self,model,test_x,test_y):
        # Train on the best model
        if self.verbose:
            print("=== MODEL ASSESSMENT ===")
        nn = Network(**model)
        nn.train(self.train_x,self.train_y)
        risk = nn.error(test_x,test_y,self.loss)
        if self.verbose:
            print(model, risk,sep='\t')
        return risk
