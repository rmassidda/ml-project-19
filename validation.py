from concurrent.futures import ProcessPoolExecutor
from grid import Grid
from network import Network
from utils import shuffle, loss_dict
import numpy as np
import functools
import time

"""
Cross-Validation
Given a model and a dataset it partitions it
by using the K-Fold CV algorithm.
It the returns the empirical risk.
"""
def cross_validation(model,x,y,loss,k=5):
    batch = np.int64(np.ceil(len(x)/k))

    """
    Each estimate uses one fold as the
    validation set and the others as
    training set, then return the number
    of epochs of training, other than the
    error on the TR and VL folds.
    """
    def estimate(i):
        rows = range(i*batch,min((i+1)*batch,len(x)-1))
        tr_x = np.delete(x,rows,0)
        tr_y = np.delete(y,rows,0)
        vl_x = x[rows]
        vl_y = y[rows]
        nn = Network(**model)
        start = time.time()
        tr_loss, vl_loss, epoch = nn.train(tr_x,tr_y,vl_x,vl_y,loss)
        end = time.time()
        return (tr_loss[0][epoch],vl_loss[0][epoch],epoch,end-start)

    """
    Using K-fold cross validation the best
    average number of epochs is found.
    Also the empirical risk is computed
    as the average on the risk per fold.
    """
    folds   = list(map(estimate,range(k)))
    tr_loss = sum([tr_loss for tr_loss,vl_loss,epoch,period in folds])/k
    vl_loss = sum([vl_loss for tr_loss,vl_loss,epoch,period in folds])/k
    epoch   = int(np.floor(sum([epoch for tr_loss,vl_loss,epoch,period in folds])/k))
    period  = sum([period for tr_loss,vl_loss,epoch,period in folds])/k

    return (tr_loss,vl_loss,epoch,period)

class Validation:
    def __init__(self,loss,workers=16,verbose=True):
        self.loss = loss
        self.executor = ProcessPoolExecutor(max_workers=workers)
        self.verbose = verbose

    """
    Model selection
    Given a data set and a dictionary of possibile
    hyperparameters it return a triple:
    - loss over the training set
    - loss over the validation set
    - the best model
    """
    def model_selection(self,hp,x,y,k):
        grid = Grid(hp)
        if self.verbose:
            print(str(k)+'-fold CV over',len(grid),'possible models')

        # Avoid order bias
        x, y = shuffle(x, y)
        
        # Parallel grid search
        p = functools.partial(cross_validation,x=x,y=y,loss=self.loss,k=k)
        res = self.executor.map(p,grid)

        model_vl   = np.Inf
        avg_period = 0
        for p, (tr_loss,vl_loss,epoch,period) in zip(grid,res):
            """
            The leaned epochs should be added only if
            the training used early stopping on the
            validation set.
            """
            if 'epochs' not in p and 'prefer_tr' in p and not p['prefer_tr']:
                p = {**p, 'epochs': epoch }

            """
            Each item in the list of results per
            hyperparameter combination is a triple
            (tr_loss,vl_loss,epoch)
            The best model is chosen according
            to the loss on the validation set
            """
            if vl_loss < model_vl:
                """
                Merge of the dictionary of hyperparameters
                with the optimal epochs.
                """
                model_tr = tr_loss
                model_vl = vl_loss
                model    = p

            if self.verbose:
                print(p,tr_loss,vl_loss,period,sep='\t')

            """
            Average time for the training
            over a fold
            """
            avg_period += period

        avg_period /= len(grid)
        if self.verbose:
            print('Average training time per fold', avg_period)

        """
        The training and the validation curves of the model are returned.
        NOTE: after the selection of the "best model" the
        model selection could proceed with a finer coarse
        grid search over smaller interval and within selected
        hyperparameters.
        """
        return (model_tr,model_vl,model)

    """
    Given a model and a dataset a neural network
    is trained using the dataset.
    The procedure returns a triple containing:
    - loss over the training set
    - loss over the test set
    - risk estimated on the test set
    """
    def estimate_test(self,model,x,y,test_x,test_y):
        # Train on the chosen model
        nn = Network(**model)
        tr_loss, te_loss, epoch = nn.train(x,y,test_x,test_y,self.loss)
        return (nn,tr_loss,te_loss,te_loss[0][epoch])

    """
    Double Cross-validation given a dataset and a
    family of functions computes the estimate of
    the risk over the family of functions.
    """
    def double_cross_validation(self,x,y,hp,k1,k2):
        x, y = shuffle(x, y)
        batch = np.int64(np.floor(len(x)/k1))

        def estimate(i):
            if self.verbose:
                print('External fold ',i)
            rows = range(i*batch,(i+1)*batch)
            tr_x = np.delete(x,rows,0)
            tr_y = np.delete(y,rows,0)
            vl_x = x[rows]
            vl_y = y[rows]
            _, _, model = self.model_selection(hp,tr_x,tr_y,k2)
            nn = Network(**model)
            nn.train(tr_x,tr_y)
            return nn.error(vl_x,vl_y,loss_dict[self.loss])

        risk = map(estimate,range(k1))
        return sum(risk)/k1
