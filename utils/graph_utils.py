import numpy as np
import scipy.sparse as sp

np.random.seed(0)


def nx_sample_neighbor(nx_g, start_node):

    neighbors = [nd for nd in nx_g.neighbors(start_node)]
    return np.random.choice(neighbors, size=1)[0]


def random_walk_with_restart(g, node_idx, restart_prob, length):
    trace = [node_idx]
    start_node = node_idx
    for _ in range(length):
        sampled_node = nx_sample_neighbor(g, start_node)
        if np.random.rand() < restart_prob:
            start_node = node_idx
        else:
            start_node = sampled_node
        trace.append(sampled_node)
    return trace


def rwr_sample_subgraph(g, node_idx, subgraph_size, max_try=5):

    neighbors = random_walk_with_restart(g, node_idx, restart_prob=1.0, length=subgraph_size*3)
    neighbors = [nd for nd in neighbors if nd>=0 and nd!=node_idx]
    neighbors = np.unique(neighbors).tolist()
    if len(neighbors) > subgraph_size - 1:
        neighbors = neighbors[:subgraph_size-1]
    else:
        retry_time = 0
        while(retry_time<max_try):
            new_nbrs = random_walk_with_restart(g, node_idx, restart_prob=0.9, length=subgraph_size*5)
            new_nbrs = [nd for nd in new_nbrs if nd>0 and nd!=node_idx]
            neighbors = np.unique(neighbors + new_nbrs).tolist()
            if len(neighbors) >= subgraph_size - 1:
                neighbors = neighbors[:subgraph_size-1]
                break
            retry_time += 1
    return [node_idx] + neighbors


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""

    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).todense()