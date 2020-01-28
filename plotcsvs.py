from matplotlib import pyplot as plt
import sys
import numpy as np

if __name__ == '__main__':

    # Command line arguments
    if len(sys.argv) > 1:
        filenames = sys.argv[1:]
    else:
        filenames = ['results/ml-cup/exp_final/cosmas_ML-CUP19-TS.csv']

    plt.title('Blind test')

    tr_csv = 'data/ML-CUP19-TR.csv'
    x  = np.genfromtxt(tr_csv,delimiter=',',usecols=range(21,22))
    y  = np.genfromtxt(tr_csv,delimiter=',',usecols=range(22,23))
    plt.plot(x, y, '.', color='red', label='Training set')

    for i, fn in enumerate(filenames):
        x = np.genfromtxt(fn, delimiter=',', usecols=range(1,2))
        y = np.genfromtxt(fn, delimiter=',', usecols=range(2,3))
        plt.plot(x, y, '.', label='Blind test '+str(i+1))

    plt.legend()
    plt.show()
    plt.cla()
    plt.clf()

    loss_csv = 'results/ml-cup/exp_final/loss.csv'

    plt.title('Risk estimation')
    epochs  = np.genfromtxt(loss_csv,delimiter=',',usecols=range(0,1))
    tr_loss = np.genfromtxt(loss_csv,delimiter=',',usecols=range(1,2))
    ts_loss = np.genfromtxt(loss_csv,delimiter=',',usecols=range(2,3))

    plt.plot(tr_loss, color="green", label='TR')
    plt.plot(ts_loss, color="blue", label='TS', linewidth=2, linestyle=':')
    plt.xlabel('Epoch')
    plt.ylabel('MEE')
    plt.legend()
    plt.grid()
    plt.show()
