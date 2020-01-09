from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from grid import Grid
from network import Network
from utils import shuffle, loss_dict
import numpy as np
import functools

"""
Hold-Out
Procedure used for model selection, given a model
and a data set, it partitions it in training and
validation set by default using 75/25% proportion.
It returns the empirical risk computed on the
validation set of a network trained on the
training set.
"""
def hold_out(model,x,y,prop=0.75):
    bound = int(len(x)*prop)        
    nn = Network(**model)
    return nn.train(x[:bound],y[:bound], x[bound:], y[bound:])

"""
Cross-Validation
Given a model and a dataset it partitions it
by using the K-Hold CV algorithm.
It the returns the empirical risk.
"""
def cross_validation(model,x,y,k=5):
    batch = np.int64(np.floor(len(x)/k))

    def estimate(i):
        rows = range(i*batch,(i+1)*batch)
        tr_x = np.delete(x,rows,0)
        tr_y = np.delete(y,rows,0)
        vl_x = x[rows]
        vl_y = y[rows]
        nn = Network(**model)
        tr_loss, vl_loss, epoch = nn.train(tr_x,tr_y,vl_x,vl_y)
        return (tr_loss,vl_loss,epoch)

    """
    Using K-hold cross validation the best
    average number of epochs is found.
    Also the empirical risk is computed
    as the average on the risk per fold.
    """
    folds = list(map(estimate,range(k)))
    # TODO: floor or ceil?
    epoch = int(np.floor(sum([epoch for tr_loss,vl_loss,epoch in folds])/k))

    """
    To plot the result of cross-validation
    the 'average' over the losses is computed
    Since the arrays from each fold have different
    lengths it's not possibile to simply sum them
    and then divide the result by k.
    It should be noticed that the only interesting
    propriety is the fact that, like the epoch is the
    average of the epochs per fold, the validation risk
    estimate should be the average over the estimates per
    fold.

    This value can be computed as
    """
    risk  = sum([vl[e] for tr,vl,e in folds])/k
    """
    Just for graphical representation motifs we propose
    the construction of an array of size [1:epoch] used 
    to construct a sound plot for the error.
    This is done by averaging the vectors via their relative position:
    """
    vl_loss = np.array([sum([vl[int(e*j/epoch)] for tr,vl,e in folds])/k for j in range(epoch+1)])
    """   
    This array is also sound for the validation process
    since it should hold:
    >>> vl_loss[epoch] == risk

    To avoid numerical issues this is forcefully constrained:
    """
    vl_loss[epoch] = risk
    """
    Same reasoning and construction is applied for
    the error in the training set.
    """
    tr_loss = np.array([sum([tr[int(e*j/epoch)] for tr,vl,e in folds])/k for j in range(epoch+1)])
    return (tr_loss,vl_loss,epoch)

class Validation:
    def __init__(self,loss,workers=16,threads=False,verbose=True):
        self.loss = loss
        if threads:
            self.executor = ThreadPoolExecutor(max_workers=workers)
        else:
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
        risk = np.Inf

        # Avoid order bias
        x, y = shuffle(x, y)
        
        # Parallel grid search
        if isinstance(self.executor,ThreadPoolExecutor):
            if k < 2:
                res = self.executor.map(lambda p: hold_out(p,x,y),grid)
            else:
                res = self.executor.map(lambda p: cross_validation(p,x,y,k),grid)
        elif isinstance(self.executor,ProcessPoolExecutor):
            if k < 2:
                p = functools.partial(hold_out,x=x,y=y)
            else:
                p = functools.partial(cross_validation,x=x,y=y)
            res = self.executor.map(p,grid)

        for p, (tr_loss,vl_loss,epoch) in zip(grid,res):
            curr_risk = vl_loss[epoch]
            if self.verbose:
                print(p,epoch,curr_risk,sep='\t')

            """
            Each item in the list of results per
            hyperparameter combination is a triple
            (tr_loss,vl_loss,epoch)
            """
            if curr_risk < risk:
                """
                Merge of the dictionary of hyperparameters
                with the optimal epochs.
                """
                best_model = {**p, 'epochs': epoch }
                best_tr    = tr_loss
                best_vl    = vl_loss
                risk = curr_risk

        """
        The training and the validation curves of the model are returned.
        NOTE: after the selection of the "best model" the
        model selection could proceed with a finer coarse
        grid search over smaller interval and within selected
        hyperparameters.
        """
        return (best_tr,best_vl,best_model)

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
        return (tr_loss,te_loss,te_loss[epoch])

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
            _, _, model = self.model_selection(hp,tr_x,tr_y,k2)
            nn = Network(**model)
            nn.train(tr_x,tr_y)
            return nn.error(vl_x,vl_y,loss_dict[self.loss])

        risk = map(estimate,range(k1))
        return sum(risk)/k1
