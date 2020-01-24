from datetime import date
from matplotlib import pyplot as plt
from random import sample
from utils import onehot
from validation import Validation
import numpy as np
import sys
import time

if __name__ == '__main__':

    # Command line arguments
    if len(sys.argv) == 4:
        bound   = float(sys.argv[1])
        k       = int(sys.argv[2])
        par_deg = int(sys.argv[3])
    else:
        print('Usage:',sys.argv[0],'test_percentage k-folds max_w',file=sys.stderr)
        print(file=sys.stderr)
        print('   test_percentage  floating number in [0,1], percentage',file=sys.stderr)
        print('                    of example to extract for the internal',file=sys.stderr)
        print('                    test set',file=sys.stderr)
        print('   k-folds          number of folds in the cross validation',file=sys.stderr)
        print('   max_w            maximum number of workers',file=sys.stderr)
        sys.exit(1)

    # File name
    train_fp = 'data/ML-CUP19-TR.csv'
    test_fp  = 'data/ML-CUP19-TS.csv'

    # Data parsing
    x       = np.genfromtxt(train_fp,delimiter=',',usecols=range(1,21))
    y       = np.genfromtxt(train_fp,delimiter=',',usecols=(21,22))
    rows    = sample(range(len(x)), k=int(len(x)*bound))
    train_x = np.delete(x,rows,0)
    train_y = np.delete(y,rows,0)
    test_x  = x[rows]
    test_y  = y[rows]

    print('Train set',len(train_x),len(train_x)/len(x),sep='\t')
    print('Test set ',len(test_x),len(test_x)/len(x),sep='\t')

    # Blind test set
    blind = np.genfromtxt(test_fp,delimiter=',',usecols=range(1,21))

    # Hyperparameters
    common = {
            'topology': [[20,30,2],[20,15,2],[20,23,2],[20,10,10,2]],
            'f_hidden': ['tanh'],
            'eta': [0.1,0.05],
            'weight_decay': [0.001,0.0001],
            'momentum': [0.9,0.99],
            'minibatch': [32],
            'max_norm': [0],
            'prefer_tr': [False],
            'patience': [100]
           }
    # Overwriting cases
    relu = {
            'f_hidden': ['relu'],
            'max_norm': [100]
            }
    decay_eta = {
            'tau': [200],
            'eta_zero': [0.5],
            'eta': [0.005]
            }

    family = [
            {**common},
            {**common, **relu},
            {**common, **decay_eta},
            {**common, **relu, **decay_eta},
            ]

    # family = [{'topology': [[20,10,2]], 'epochs': [2,5,20]}]

    # Validation
    val = Validation(['MEE'],workers=par_deg,verbose=True)
    start = time.time()

    # Select the best model via cross-validation
    print('Model selection')
    model_tr, model_vl, model = val.model_selection(family,train_x,train_y,k)
    print('Chosen model:')
    print(model,model_tr,model_vl,end='\n\n')

    ms_time = time.time()
    print('Time for model selection: %f' % (ms_time-start))

    # Model assessment on the test set
    print('Model assessment')
    nn, tr_loss, ts_loss, risk  = val.estimate_test(model,train_x,train_y,test_x,test_y)
    print('Chosen model:')
    print(model,risk)

    end = time.time()
    print('Time for model assessment: %f' % (end-ms_time))
    print('Total time elapsed: %f' % (end-start))
    print('Start plotting')

    # Plot of the estimation
    plt.title('Risk estimation')
    plt.plot(tr_loss[0], color="green", label='TR')
    plt.plot(ts_loss[0], color="blue", label='TS', linewidth=2, linestyle=':')
    plt.xlabel('Epoch')
    plt.ylabel('MEE')
    plt.legend()
    plt.savefig('risk_estimation.png')

    # Results for the blind test
    with open('rottenmeier_ML-CUP19-TS.csv', 'w+') as fp:
        print('# Emanuele  Cosenza	Riccardo Massidda',file=fp)
        print('# rottenmeier',file=fp)
        print('# ML-CUP19',file=fp)
        print('# '+date.today().strftime("%d/%m/%Y"),file=fp)
        for i in range(len(blind)):
            out = nn.predict(blind[i])
            print(i+1,out[0],out[1],sep=',',file=fp)
