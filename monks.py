from matplotlib import pyplot as plt
from utils import onehot
from validation import Validation
import numpy as np
import sys
import time

if __name__ == '__main__':

    # Command line arguments
    if len(sys.argv) == 4:
        train_fp = sys.argv[1]
        test_fp  = sys.argv[2]
        par_deg  = int(sys.argv[3])
    else:
        train_fp = 'data/monks/monks-1.train'
        test_fp  = 'data/monks/monks-1.test'
        par_deg  = 8

    monk_ranges = [3,3,2,3,4,2]

    # Data load
    raw_x = np.genfromtxt(train_fp,usecols=(1,2,3,4,5,6))
    train_x = np.array(list(map(lambda x: onehot(x,monk_ranges),raw_x)))
    train_y = np.genfromtxt(train_fp,usecols=(0))
    raw_x = np.genfromtxt(test_fp,usecols=(1,2,3,4,5,6))
    test_x = np.array(list(map(lambda x: onehot(x,monk_ranges),raw_x)))
    test_y = np.genfromtxt(test_fp,usecols=(0))

    # Hyperparameters
    early_stopping = {
        'f_hidden' : ['tanh'],
        'f_output' : ['sigmoid'],
        'minibatch': [1,16],
        'eta': [1e-1,1e-2],
        'topology': [[17,2,1],[17,4,1],[17,8,1]],
        'momentum': [0.9,0.99],
        'weight_decay': [1e-2,1e-4],
        'patience': [50]
        }
    fixed_epoch = {
        'f_hidden' : ['tanh'],
        'f_output' : ['sigmoid'],
        'minibatch': [1,16],
        'eta': [1e-1,1e-2],
        'topology': [[17,2,1],[17,4,1],[17,8,1]],
        'momentum': [0.9,0.99],
        'weight_decay': [1e-2,1e-4],
        'epochs': [500]
        }

    lite = {
        'topology': [[17,2,1],[17,4,1],[17,8,1]],
    }

    family = [early_stopping, fixed_epoch]
    #NOTE: uncomment for lite test
    # family = [lite]

    # Validation
    val = Validation(['MCL','MSE'],workers=par_deg,verbose=True)

    start = time.time()

    # Select the best model via cross-validation
    print('Model selection')
    model_tr, model_vl, model = val.model_selection(family,train_x,train_y,5)
    print('Chosen model:')
    print(model,model_tr,model_vl,end='\n\n')

    # Model assessment on the test set
    print('Model assessment')
    _, tr_err, ts_err, risk  = val.estimate_test(model,train_x,train_y,test_x,test_y)
    print('Chosen model:')
    print(model,risk)

    end = time.time()
    print('Total time elapsed: %f' % (end-start))

    # Plot of the estimation
    tr_accs = 100 * (1 - tr_err[0])
    ts_accs = 100 * (1 - ts_err[0])
    plt.title('MCL Monks')
    plt.plot(tr_accs, color="green", label='TR')
    plt.plot(ts_accs, color="blue", label='TS', linewidth=2, linestyle=':')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig('MCL_monks.png')
    
    plt.cla()
    plt.clf()

    plt.title('MSE Monks')
    plt.plot(tr_err[1], color="green", label='TR')
    plt.plot(ts_err[1], color="blue", label='TS', linewidth=2, linestyle=':')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig('MSE_monks.png')
