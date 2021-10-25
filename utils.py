import numpy as np
import igraph as ig
import networkx as nx

def compute_caus_order(G):
    d = G.shape[0]
    remain = list(range(d))
    caus_order = np.empty(d, dtype = int)
    for i in range(d-1):
        root = min(np.where(G.sum(axis=0) == 0)[0])
        caus_order[i] = remain[root]
        del remain[root]
        G = np.delete(G, root, axis = 0)
        G = np.delete(G, root, axis = 1)
    caus_order[d-1] = remain[0]
    return caus_order

def find_pa(G, node):
    return np.where(G[:,node] == 1)[0]

def test_order(est_order, G):
    ed_from, ed_to = np.where(G==1)
    order_index = np.argsort(est_order)
    count = 0
    for i in range(len(ed_from)):
        if order_index[ed_from[i]] > order_index[ed_to[i]]:
            count += 1
    return int(count==0), count


def simulate_dag(d, s0, graph_type, permute=True):
    '''Simulate random DAG with some expected number of edges.

    Parameters
    ----------
        d : int
            num of nodes
        s0 : int
            expected num of edges
        graph_type : str
            'ER', 'SF', 'Tree', 'MC'

    Returns
    ----------
    B : np.array
        binary adj matrix of DAG
    '''
    max_num_edge = d * (d - 1) / 2
    if graph_type == 'ER':
        # Erdos-Renyi
        edge_from, edge_to = np.nonzero(np.triu(np.ones(d), k = 1))
        edges = np.random.choice(len(edge_from), min(s0, max_num_edge), replace = False)
        edge_from = edge_from[edges]
        edge_to = edge_to[edges]
        B = np.zeros((d, d))
        B[edge_from, edge_to] = 1
        if permute:
            rand_sort = np.arange(d)
            np.random.shuffle(rand_sort)
            B = B[rand_sort, :]
            B = B[:, rand_sort]
        
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(min(s0, max_num_edge) / d)), directed=True)
        B = np.array(G.get_adjacency().data)

    elif graph_type == 'Tree':
        # Tree graph
        B = np.tril(nx.to_numpy_matrix(nx.generators.trees.random_tree(d)))

    elif graph_type == 'MC':
        # Markov chain
        B = np.eye(d, k = 1)
        if permute:
            rand_sort = np.arange(d)
            np.random.shuffle(rand_sort)
            B = B[rand_sort, :]
            B = B[:, rand_sort]

    return B

  
def sample_from_bino_addtive(G, n, prob):
    d = G.shape[0]
    X = np.empty((n,d))
    caus_order = compute_caus_order(G)
    for node in caus_order:
        pa_of_node = np.where(G[:,node] == 1)[0]
        if len(pa_of_node) == 0:
            X[:,node] = np.random.binomial(1, prob, n)
        else:
            X[:,node] = np.random.binomial(1, prob, n) + X[:,pa_of_node].sum(axis=1)
    return X

  
def sample_from_mod(G, n, prob, shuffle=True):
  d = G.shape[0]
  X = np.empty((n,d))
  caus_order = compute_caus_order(G)
  for node in caus_order:
      pt = np.random.choice([prob, 1-prob]) if shuffle else prob
      pa_of_node = np.where(G[:,node] == 1)[0]
      if len(pa_of_node) == 0:
          X[:,node] = np.random.binomial(1, pt, n)
      else:
          S = X[:,pa_of_node].sum(axis=1)
          Yt = np.mod(S, 2)
          ind = np.random.binomial(1, pt, n)
          X[:,node] = (Yt ** ind) * ((1-Yt) ** (1-ind))
  return X


def count_accuracy(B_true, B_est):
    """Compute various accuracy metrics for B_est.

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size}
