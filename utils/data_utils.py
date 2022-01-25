import os
import sys
import numpy as np
import scipy.sparse as sp
import pickle as pkl
import networkx as nx
import scipy.io as sio
from sklearn import preprocessing


def dense_to_sparse(dense_matrix):
    shape = dense_matrix.shape
    row = []
    col = []
    data = []
    for i, r in enumerate(dense_matrix):
        for j in np.where(r > 0)[0]:
            row.append(i)
            col.append(j)
            data.append(dense_matrix[i,j])

    sparse_matrix = sp.coo_matrix((data, (row, col)), shape=shape).tocsc()
    return sparse_matrix


def load_citation_datadet(path, dataset_str):

    dataset_path = os.path.join(path, dataset_str)
    def parse_index_file(filename):
        """Parse index file."""
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    names = [f'ind.{dataset_str}.{name}' for name in names]
    objects = []
    for name in names:
        with open(os.path.join(dataset_path, name), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(dataset_path, f'ind.{dataset_str}.test.index'))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    adj_dense = np.array(adj.todense(), dtype=np.float64)
    attribute_dense = np.array(features.todense(), dtype=np.float64)
    cat_labels = np.array(np.argmax(labels, axis = 1).reshape(-1,1), dtype=np.uint8)

    return attribute_dense, adj_dense, cat_labels


def load_ad_dataset(path, dataset_str):

    data_path = os.path.join(path, dataset_str)
    data = sio.loadmat(os.path.join(data_path, f'{dataset_str}.mat'))
    attribute_dense = np.array(data['Attributes'].todense())
    attribute_dense = preprocessing.normalize(attribute_dense, axis=0)
    adj_dense = np.array(data['Network'].todense())
    cat_labels = data['Label']
    return attribute_dense, adj_dense, cat_labels


def save_dgl_fraud_data(dataset="amazon", path="./"):
    from dgl.data.fraud import FraudDataset
    data = FraudDataset(dataset)
    graph = data[0]
    feat = graph.ndata['feature']
    label = graph.ndata['label']
    adjs = [graph.adj(scipy_fmt='coo', etype=etp) for etp in graph.etypes]

    data = {
        "feat": dense_to_sparse(feat),
        "label": label,
        "adj": adjs
    }
    print(f"Number of nodes: {label.shape[0]}",
    )
    sio.savemat(data, path + f"{dataset}.mat")


def shuffled_index(batch_size, graph_size):
    feat = np.zeros((batch_size, graph_size, graph_size))
    for i in range(batch_size):
        feat[i, Ellipsis] = np.eye(graph_size)
        np.random.shuffle(feat[i, 1:, :])
    return feat

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

# from sklearn import preprocessing
# def preprocess_features(features):
#     return preprocessing.StandardScaler().fit_transform(features.todense())