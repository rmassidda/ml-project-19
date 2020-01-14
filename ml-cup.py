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
        train_fp = sys.argv[1]
        test_fp  = sys.argv[2]
        par_deg  = int(sys.argv[3])
    else:
        train_fp = 'data/ML-CUP19-TR.csv'
        test_fp  = 'data/ML-CUP19-TS.csv'
        par_deg  = 8

    # Data parsing
    x       = np.genfromtxt(train_fp,delimiter=',',usecols=range(1,21))
    y       = np.genfromtxt(train_fp,delimiter=',',usecols=(21,22))
    rows    = sample(range(len(x)), k=int(len(x)/4))
    train_x = np.delete(x,rows,0)
    train_y = np.delete(y,rows,0)
    test_x  = x[rows]
    test_y  = y[rows]

    print('Train set',len(train_x),len(train_x)/len(x),sep='\t')
    print('Test set',len(test_x),len(test_x)/len(x),sep='\t')

    # Blind test set
    blind = np.genfromtxt(test_fp,delimiter=',',usecols=range(1,21))

    # Hyperparameters
    common = {
        'topology': [[20,30,2],[20,15,2],[20,15,10,2]],
        'momentum': [0.9,0.99],
        'weight_decay': [1e-2,1e-4],
        'eta': [1e-1,1e-2],
        'minibatch': [16,32,64],
        'f_hidden': ['tanh'],
        'f_output': ['identity']
        }
    early_stopping = {**common, 'patience': [50] }
    gradient_stop  = {**common, 'tol': [10] }
    lite = {
        'topology': [[20,30,2],[20,15,2],[20,15,10,2]],
    }

    family = [early_stopping, gradient_stop]
    #NOTE: uncomment for lite test
    # family = [lite]

    # Validation
    val = Validation(['MEE'],workers=par_deg,verbose=True)
    start = time.time()

    # Select the best model via cross-validation
    print('Model selection')
    model_tr, model_vl, model = val.model_selection(family,train_x,train_y,5)
    print('Chosen model:')
    print(model,model_tr,model_vl,end='\n\n')

    # Model assessment on the test set
    print('Model assessment')
    nn, tr_loss, ts_loss, risk  = val.estimate_test(model,train_x,train_y,test_x,test_y)
    print('Chosen model:')
    print(model,risk)

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

    end = time.time()
    print('Total time elapsed: %f' % (end-start))
