from concurrent.futures import ProcessPoolExecutor
from matplotlib import pyplot as plt
from network import Network
from validation import Validation, cross_validation
from grid import Grid
from random import sample
import numpy as np
import functools
import sys

class Experiment:
    def __init__(self,name,models):
        self.name = name
        self.grid = Grid(models)

    def __len__(self):
        return len(self.grid)

"""
Ad-hoc initial values, for each experiment
these are differently overwritten
"""
init = {
        'topology': [[20,15,2]],
       }

"""
Trains a neural network over a given
dataset and return the learning curve
"""
def worker(seed,m,x,y):
    # Set seed for the current process
    np.random.seed(seed)
    nn = Network(**m)
    return nn.train(x,y)

"""
Given a number of experiments, compute the average of the
results and output the relative plots.
"""
if __name__ == '__main__':

    if len(sys.argv) == 6:
        n_trials = int(sys.argv[1])
        max_w    = int(sys.argv[2])
        epoch    = int(sys.argv[3])
        k        = int(sys.argv[4])
        coverage = float(sys.argv[5])
    else:
        print('Usage:',sys.argv[0],'n_trials max_w epoch k coverage',file=sys.stderr)
        sys.exit(1)

    # File name
    train_fp = 'data/ML-CUP19-TR.csv'

    # Data parsing
    x       = np.genfromtxt(train_fp,delimiter=',',usecols=range(1,21))
    y       = np.genfromtxt(train_fp,delimiter=',',usecols=(21,22))

    # Sampling
    rows    = sample(range(len(x)), k=int(len(x)*coverage))
    train_x  = x[rows]
    train_y  = y[rows]

    # Parallel executor
    executor = ProcessPoolExecutor(max_workers=max_w)

    # Define experiments
    experiments = []
    experiments.append(Experiment('Fixed learning rate', [{**init, 'eta': [1,5e-1,1e-1,5e-2,1e-2,1e-3,1e-4,1e-5], 'epochs': [epoch]}]))
    experiments.append(Experiment('Decay learning rate', [{**init, 'eta_zero': [5e-1], 'eta': [5e-3], 'tau': [100,200], 'epochs': [epoch]},
        {**init, 'eta_zero': [1e-1], 'eta': [1e-3], 'tau': [100,200], 'epochs': [epoch]}]))
    experiments.append(Experiment('Oscillating decay learning rate', [{**init, 'eta_zero': [5], 'eta': [5e-2], 'tau': [100,200], 'epochs': [epoch]},
        {**init, 'eta_zero': [1], 'eta': [1e-2], 'tau': [100,200], 'epochs': [epoch]}]))
    experiments.append(Experiment('Minibatch-bigstep', [{**init, 'eta': [1,0.5], 'minibatch': [1,4], 'epochs': [epoch]}]))
    experiments.append(Experiment('Tikhonov regularization (L2)', [{**init, 'weight_decay': [0,1e-1,1e-2,1e-3,1e-4], 'epochs': [epoch]}]))
    experiments.append(Experiment('Momentum', [{**init, 'momentum': [0,0.5,0.9,0.99,0.999], 'epochs': [epoch]}]))
    experiments.append(Experiment('Gradient clipping', [{**init, 'max_norm': [1,2,10,100], 'epochs': [epoch]}]))
    experiments.append(Experiment('Activation functions', [{**init, 'f_hidden': ['tanh','sigmoid','relu'], 'epochs': [epoch]}]))

    # Status
    print('Train set',len(train_x),len(train_x)/len(x),sep='\t')
    print('Run of',len(experiments),'experiments')

    # Execute experiments
    counter = 1
    for exp in experiments:
        # Initialize
        print('\n"'+exp.name+'"','consists of',len(exp.grid),'models')
        plt.title(exp.name+' MSE')
        loss = [np.zeros(epoch+1) for i in range(len(exp.grid))]

        # Compute tests
        for i,m in zip(range(len(exp.grid)),exp.grid):
            # Map
            w = functools.partial(worker,
                    m=m,
                    x=train_x,
                    y=train_y)
            seeds = np.random.randint(2**32, size=n_trials)
            res = executor.map(w,seeds)

            # Reduce
            for (tr_err,ts_err,e) in res:
                # Update accuracy
                loss[i] += tr_err[0]
            loss[i] /= n_trials

            # Output
            print(i,m,loss[i][epoch])
            plt.plot(loss[i], label='Model '+str(i))

        # Plot
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        plt.savefig('screening_'+str(counter)+'.png', dpi=300)

        # Get ready for next experiment
        counter += 1
        plt.cla()
        plt.clf()

    # Special case to assess the patience
    name = 'Assess different values of patience,'
    val  = Validation(['MEE'],workers=max_w)
    pat  = [1,2,4,8,16,32,64,128,256,512]
    lss  = []
    eph  = []
    hp   = [{**init, 'prefer_tr': [False], 'patience': pat}]
    grid = Grid(hp)

    print('\n"'+name+'"','consists of',len(grid),'models')
    print(str(k)+'-fold cross validation')

    p = functools.partial(cross_validation,x=train_x,y=train_y,loss=['MSE'],k=k)
    res = executor.map(p,grid)
    for p, (tr_loss,vl_loss,epoch, period) in zip(grid,res):
        p = {**p, 'epochs': epoch }
        print(p,tr_loss,vl_loss,period,sep='\t')
        lss.append(vl_loss)
        eph.append(epoch)

    plt.title(name+' validation MSE')
    plt.plot(pat, lss, label='Model '+str(counter))
    plt.xlabel('Patience')
    plt.ylabel('MSE')
    plt.savefig('screening_'+str(counter)+'.png', dpi=300)
    counter += 1
    plt.cla()
    plt.clf()
    plt.title(name+' epochs')
    plt.plot(pat, eph, label='Model '+str(counter))
    plt.xlabel('Patience')
    plt.ylabel('epochs')
    plt.savefig('screening_'+str(counter)+'.png', dpi=300)
