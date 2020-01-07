from concurrent.futures import ThreadPoolExecutor
from grid import Grid
from network import Network
from utils import onehot, shuffle
import numpy as np
import sys

class Validation:
    def __init__(self,loss,workers=16,verbose=True):
        self.loss = loss
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self.verbose = verbose

    """
    Hold-Out
    Procedure used for model selection, given a model
    and a data set, it partitions it in training and
    validation set by default using 75/25% proportion.
    It returns the empirical risk computed on the
    validation set of a network trained on the
    training set.
    """
    def hold_out(self,model,x,y,prop=0.75):
        bound = int(len(x)*prop)        
        nn = Network(**model)
        tr_loss, vl_loss, epoch = nn.train(x[:bound],y[:bound], x[bound:], y[bound:])
        return (vl_loss[epoch], epoch)

    """
    Cross-Validation
    Given a model and a dataset it partitions it
    by using the K-Hold CV algorithm.
    It the returns the empirical risk.
    """
    def cross_validation(self,model,x,y,k=5):
        batch = np.int64(np.floor(len(x)/k))

        def estimate(i):
            rows = range(i*batch,(i+1)*batch)
            tr_x = np.delete(x,rows,0)
            tr_y = np.delete(y,rows,0)
            vl_x = x[rows]
            vl_y = y[rows]
            nn = Network(**model)
            tr_loss, vl_loss, epoch = nn.train(tr_x,tr_y,vl_x,vl_y)
            return (vl_loss[epoch], epoch)

        folds = list(map(estimate,range(k)))
        error = sum([loss for loss,epoch in folds])/k
        epoch = round(sum([epoch for loss,epoch in folds])/k)
        return (error, epoch)

    """
    Model selection
    Given a data set and a dictionary of possibile
    hyperparameters it return the best model
    """
    def model_selection(self,hp,x,y,k):
        grid = Grid(hp)
        risk = np.Inf
        best = None

        # Avoid order bias
        x, y = shuffle(x, y)
        
        # Parallel grid search
        if k < 2:
            res = self.executor.map(lambda p: self.hold_out(p,x,y),grid)
        else:
            res = self.executor.map(lambda p: self.cross_validation(p,x,y,k),grid)

        for p, r in zip(grid,res):
            if self.verbose:
                print(p,r,sep='\t')

            """
            Each item in the list of results per
            hyperparameter combination is a tuple
            where the first element is the error
            on the validation set, while the other
            is the number of optimal epochs chose
            via early stopping.
            r[0]     validation error
            r[1]     epoch
            """
            if r[0] < risk:
                """
                Merge of the dictionary of hyperparameters
                with the optimal epochs.
                """
                best = {**p, 'epochs': r[1] }
                risk = r[0]

        """
        NOTE: after the selection of the "best model" the
        model selection could proceed with a finer coarse
        grid search over smaller interval and within sele
        cted hyperparameters
        """
        return best

    """
    Given a model and a dataset a neural network
    is trained using the dataset.
    The procedure returns the risk computed
    on the test set.
    """
    def estimate_test(self,model,x,y,test_x,test_y):
        # Train on the chosen model
        nn = Network(**model)
        nn.train(x,y)
        return nn.error(test_x,test_y,self.loss)

    """
    Double Cross-validation given a dataset and a
    family of functions computes the estimate of
    the risk over the family of functions.
    """
    def double_cross_validation(self,x,y,hp,k1,k2):
        batch = np.int64(np.floor(len(x)/k1))

        def estimate(i):
            rows = range(i*batch,(i+1)*batch)
            tr_x = np.delete(x,rows,0)
            tr_y = np.delete(y,rows,0)
            vl_x = x[rows]
            vl_y = y[rows]
            model = self.model_selection(hp,tr_x,tr_y,k2)
            nn = Network(**model)
            nn.train(tr_x,tr_y)
            return nn.error(vl_x,vl_y,self.loss)

        risk = map(estimate,range(k1))
        return sum(risk)/k1
