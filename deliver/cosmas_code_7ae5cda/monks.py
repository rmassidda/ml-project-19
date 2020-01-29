from concurrent.futures import ProcessPoolExecutor
from matplotlib import pyplot as plt
from network import Network
from utils import onehot
import functools
import numpy as np
import sys

def worker(seed,m,x,y,tx,ty,l):
    # Set seed for the current process
    np.random.seed(seed)
    nn = Network(**m)
    return nn.train(x,y,tx,ty,l)

"""
Given a number of experiments, compute the average of the
results and output the relative plots.
"""
if __name__ == '__main__':

    if len(sys.argv) == 4:
        n_trials = int(sys.argv[1])
        max_w    = int(sys.argv[2])
        epoch    = int(sys.argv[3])
    else:
        print('Usage:',sys.argv[0],'n_trials max_w epoch',file=sys.stderr)
        sys.exit(1)

    common = {
        'f_hidden': 'tanh',
        'f_output': 'sigmoid',
        'minibatch': 32,
        'eta': 0.5,
        'momentum': 0,
        'weight_decay': 0,
        'topology': [17,4,1],
        'epochs': epoch,
    }

    monk_ranges = [3,3,2,3,4,2]

    model = [
            { **common},
            { **common},
            { **common},
            { **common, 'topology': [17,4,1], 'weight_decay': 1e-2 }
            ]

    dataset  = ['monks-1','monks-2','monks-3','monks-3']
    name     = ['monks-1','monks-2','monks-3','monks-3_reg']
    losses   = ['MCL', 'MSE']

    print('Running',n_trials,'trials for',epoch,'epochs')
    print('Using',max_w,'parallel workers')

    for i in range(len(dataset)):
        # Read from file
        train_fp = 'data/monks/'+dataset[i]+'.train'
        test_fp  = 'data/monks/'+dataset[i]+'.test'

        # Training set
        raw_x = np.genfromtxt(train_fp,usecols=(1,2,3,4,5,6))
        train_x = np.array(list(map(lambda x: onehot(x,monk_ranges),raw_x)))
        train_y = np.genfromtxt(train_fp,usecols=(0))

        # Test set
        raw_x = np.genfromtxt(test_fp,usecols=(1,2,3,4,5,6))
        test_x = np.array(list(map(lambda x: onehot(x,monk_ranges),raw_x)))
        test_y = np.genfromtxt(test_fp,usecols=(0))

        # Average results
        avg_acc_tr = np.zeros(epoch+1)
        avg_acc_ts = np.zeros(epoch+1)
        avg_mse_tr = np.zeros(epoch+1)
        avg_mse_ts = np.zeros(epoch+1)

        # Map phase
        if max_w == 0:
            # Sequential experiments
            res = []
            for j in range(n_trials):
                res.append(worker(j,model[j],train_x,train_y,test_x,test_y,losses))
        else:
            # Parallel experiments
            w = functools.partial(worker,
                    m=model[i],
                    x=train_x,
                    y=train_y,
                    tx=test_x,
                    ty=test_y,
                    l=losses)

            executor = ProcessPoolExecutor(max_workers=max_w)
            seeds = np.random.randint(2**32, size=n_trials)
            res = executor.map(w,seeds)

        # Reduce results
        for (tr_err,ts_err,e) in res:
            # Update accuracy
            avg_acc_tr += (100 * (1 - tr_err[0]))
            avg_acc_ts += (100 * (1 - ts_err[0]))

            # Update MSE
            avg_mse_tr += tr_err[1]
            avg_mse_ts += ts_err[1]

        avg_acc_tr /= n_trials 
        avg_acc_ts /= n_trials
        avg_mse_tr /= n_trials
        avg_mse_ts /= n_trials

        # Print results
        print(name[i],model[i])
        print('Accuracy:',avg_acc_tr[epoch],avg_acc_ts[epoch])
        print('MSE:',avg_mse_tr[epoch],avg_mse_ts[epoch])

        # Plot of the estimation
        plt.title('MCL Monks')
        plt.plot(avg_acc_tr, color="green", label='TR')
        plt.plot(avg_acc_ts, color="blue", label='TS', linewidth=2, linestyle=':')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.savefig(name[i]+'_MCL.png')
        plt.cla()
        plt.clf()

        plt.title('MSE Monks')
        plt.plot(avg_mse_tr, color="green", label='TR')
        plt.plot(avg_mse_ts, color="blue", label='TS', linewidth=2, linestyle=':')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        plt.savefig(name[i]+'_MSE.png')
        plt.cla()
        plt.clf()
