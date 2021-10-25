import numpy as np
from entropy import *

def est_entropy(x, A = None):
    '''helper function for estimating (conditional) entropy H(x|A).
    Parameters
    ----------
        x : np.array
            Data of the R.V. to be estimate
        A : 2d np.array
            Data matrix of conditioning set
            Can be None, then return marginal entropy
    Returns
    ----------
        h : float
            estimated (conditional) entropy
    '''    
    A = A if (A is None) or (len(A.shape) == 2) else A[:,np.newaxis]
    if len(np.unique(x)) == 1:
        return 0
    elif (A is None) or (A.shape[1] == 0):
        k, counts = np.unique(x, return_counts = True)
        k = min(len(k), 100000)
        fin = hist_to_fin(counts)
        entropy = Entropy(k = k)
        entro = entropy.estimate(fin)*np.log(2)
        return entro
    else:
        logk = np.apply_along_axis(lambda y: np.log(len(np.unique(y))), 0, A).sum()
        k = np.round(np.exp(min(logk, np.log(100000))))
        if k == 1:
            partial = 0
        else:
            counts = np.unique(A, return_counts = True, axis=0)[1]
            fin = hist_to_fin(counts)
            entropy = Entropy(k = k)
            partial = entropy.estimate(fin)*np.log(2)

        logk += np.log(len(np.unique(x)))
        k = np.round(np.exp(min(logk, np.log(100000))))
        counts = np.unique(np.c_[x, A], return_counts = True, axis=0)[1]
        fin = hist_to_fin(counts)
        entropy = Entropy(k = k)
        total = entropy.estimate(fin)*np.log(2)

        return total - partial

    
def findPPS(x, A, kappa):

    '''PPS procedure to find Markov boundary.
    Parameters
    ----------
        x : np.array
            Data of the R.V. to be estimate
        A : 2d np.array
            Data matrix of candidate set
        kappa : float
            PPS threshold
    Returns
    ----------
        pps : np.array
            Index of estimated Markov boundary in A
    '''    
    
    if A.size == 0:
        return np.array([], dtype=int)
    pps = []
    entro_now = est_entropy(x)
    rest_pps = np.arange(A.shape[1])
    while True:
        entro_cond = np.array([est_entropy(x, A = np.c_[A[:,pps], A[:,pp]]) for pp in rest_pps])
        mutInfor = entro_now - entro_cond
        if max(mutInfor) < kappa:
            break
        pps.append(rest_pps[np.argmax(mutInfor)])
        entro_now = entro_cond[np.argmax(mutInfor)]
        rest_pps = np.setdiff1d(np.arange(A.shape[1]), pps)
        if len(rest_pps) == 0:
            break
    return np.array(pps, dtype=int)


class TAM():
    def __init__(self, X):
        
        '''Main class of TAM DAG learning algorithm.
        Parameters
        ----------
            X : 2d np.array
                data matrix
        '''
        
        self.X = X
        self.n, self.d = X.shape   
        self.ancestor = np.array([], dtype=int)
        self.descendant = np.arange(self.d)
        self.layers = []
    
    def train(self, kappa, omega):
        
        '''Run the TAM algorithm to learn the DAG.
        Parameters
        ----------
            kappa : float
                PPS threshold
            omega : float
                Mutual information testing threshold
        Returns
        ----------
            self.Gr : np.array
                binary adj matrix of DAG
            self.layers : list of np.array
                layer decomposition of DAG
        '''
        
        self.Gr = np.zeros((self.d, self.d))
        while len(self.descendant) > 0:
            
            # call PPS() to find Markov boundary and estimate conditional entropy
            ppss = {}
            condentro = []
            for j in self.descendant:
                pps = findPPS(self.X[:,j], self.X[:,self.ancestor], kappa)
                ppss[j] = self.ancestor[pps]
                condentro.append(est_entropy(self.X[:,j], self.X[:,ppss[j]]))
            condentro = np.array(condentro)
            tau = self.descendant[np.argsort(condentro)]
            condentro = np.sort(condentro)
            
            # TAM step
            cond = np.array([], dtype=int)
            mask = np.array([], dtype=int)
            
            while len(tau) > 0:
                cond = np.r_[cond, tau[0]]
                condentro = condentro[1:]
                tau = tau[1:]           
                # testing
                if len(tau) == 0:
                    break
                condentro2 = []
                for j in tau:
                    condentro2.append(est_entropy(self.X[:,j], self.X[:, np.r_[ppss[j], cond]]))
                condentro2 = np.array(condentro2)
                # masking
                index = (condentro - condentro2 > omega)
                mask = np.r_[mask, tau[index]]
                condentro = condentro[~index]
                tau = tau[~index]
            
            # update layers and graphs
            self.layers.append(cond)
            self.ancestor = np.r_[self.ancestor, cond]
            self.descendant = np.setdiff1d(self.descendant, self.ancestor)
            for j in cond:
                self.Gr[ppss[j], j] = 1
