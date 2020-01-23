from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    plt.title('Blind test')
    
    fn = 'data/ML-CUP19-TR.csv'
    x  = np.genfromtxt(fn,delimiter=',',usecols=range(21,22))
    y  = np.genfromtxt(fn,delimiter=',',usecols=range(22,23))
    plt.plot(x, y, '.', color='red', label='Training set')

    fn = 'results/ml-cup/rottenmeier_ML-CUP19-TS.csv'
    x  = np.genfromtxt(fn,delimiter=',',usecols=range(1,2))
    y  = np.genfromtxt(fn,delimiter=',',usecols=range(2,3))
    plt.plot(x, y, '.', color='green', label='Blind test')

    fn = 'rottenmeier_ML-CUP19-TS.csv'
    try:
        x  = np.genfromtxt(fn,delimiter=',',usecols=range(1,2))
        y  = np.genfromtxt(fn,delimiter=',',usecols=range(2,3))
        plt.plot(x, y, '.', color='blue', label='Last blind test')
    except OSError:
        pass

    plt.legend()
    plt.show()
