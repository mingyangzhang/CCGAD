import os

import random
import argparse

import numpy as np
import scipy.io as sio
import scipy.sparse as sp


parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='./raw_dataset/')

parser.add_argument('--dataset', type=str, default='BlogCatalog')
parser.add_argument('--n', type=int, default=50)  #num of anomalies
parser.add_argument('--dir', type=str, default='dataset/')  #'BlogCatalog'  'Flickr' 'cora'  'citeseer'  'pubmed'

args = parser.parse_args()

data_path = args.data_path
dataset_str = args.dataset

# Set seed
np.random.seed(0)
random.seed(0)

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

### Load data ###
# data is a dict with 3 keys:
# - adj: n*n, sparse adj matrix
# - feats: n*d, sparse feature matrix
# - class: n, indicate class of every node

print(f'Loading data: {dataset_str}')

data = sio.loadmat(os.path.join(data_path, dataset_str) + ".mat")

adj = data["adj"]
feat_dense = data["feats"].todense()
cat_labels = data["class"]

num_node = cat_labels.shape[0]

# Random pick anomaly nodes
all_idx = list(range(num_node))
random.shuffle(all_idx)
anomaly_idx = all_idx[:args.n]
label = np.zeros((num_node, 1), dtype=np.uint8)


label[anomaly_idx] = 1

print('Constructing anomaly nodes...')
for i in range(args.n):
    anomaly_cat = cat_labels[anomaly_idx[i]]
    candidate_idxs = np.where(cat_labels!=anomaly_cat)[0]
    np.random.shuffle(candidate_idxs)
    rand_idx = candidate_idxs[0]
    feat_dense[anomaly_idx[i], :] = feat_dense[rand_idx, :]


# Pack & save them into .mat
print('Saving mat file...')
feats = dense_to_sparse(feat_dense)

if not os.path.exists(args.dir):
    os.makedirs(args.dir)

save_name = os.path.join(f'{args.dir}', f'{dataset_str}.mat')
sio.savemat(save_name, {'adj': adj, 'label': label, 'feats': feats, 'class':cat_labels})

print(f'Done. The file is save as: {save_name}')
