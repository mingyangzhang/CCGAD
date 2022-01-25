import numpy as np
import networkx as nx
import scipy.io as sio
from tqdm import tqdm

from utils.graph_utils import rwr_sample_subgraph
from utils.data_utils import preprocess_features
from sklearn import preprocessing


class GraphLoader(object):
    """  """
    def __init__(self, args):
        self.args = args

        graph = sio.loadmat(args.graph_path + args.dataset + ".mat")

        self.adj = graph['adj'].todense()
        self.n_nodes = self.adj.shape[0]

        self.adj = self.adj + np.eye(self.n_nodes)

        self.nx_g = nx.from_numpy_matrix(self.adj)

        # self.feats = preprocess_features(graph['feats'])
        # feat_dense = self.feats.todense()

        self.feats = graph['feats']
        feat_dense = self.feats.todense()
        feat_dense = preprocessing.StandardScaler().fit_transform(feat_dense)


        self.feat_padded = np.pad(feat_dense, ((1, 0), (0, 0)), 'constant')
        self.adj_padded = np.pad(self.adj, ((1, 0), (1, 0)), 'constant')

        self.abnr_labels = np.reshape(graph['label'], (-1, 1))
        self.class_labels = np.reshape(graph['class'], (-1, 1))

        self.subg_size = args.subgraph_size
        if args.rw_sample == "resample":
            self.subgs = self.sample_subgraphs(np.arange(self.n_nodes))
            np.save(f"outputs/{args.dataset}_subgs.npy", self.subgs)
        else:
            self.subgs = np.load(f"outputs/{args.dataset}_subgs.npy")

    def sample_subgraphs(self, node_idxs):
        subgs = []
        pbar = tqdm(node_idxs)
        pbar.set_description("RW sampling...")
        for node_idx in pbar:
            subg_nodes = rwr_sample_subgraph(self.nx_g, node_idx, self.args.subgraph_size)
            if len(subg_nodes) < self.subg_size:
                subg_nodes = subg_nodes + [-1]*(self.subg_size - len(subg_nodes))
            subgs.append(subg_nodes)
        subgs = np.array(subgs)
        return subgs

    def batch_data(self, batch):

        node_idxs = batch["node_idxs"]
        n = len(node_idxs)
        subg_node_idxs = self.subgs[node_idxs]
        subg_feats = self.feat_padded[subg_node_idxs + 1]
        subg_adjs = []
        for i in range(n):
            adj = self.adj_padded[np.ix_(subg_node_idxs[i] + 1, subg_node_idxs[i] + 1)]
            adj_norm = adj / (np.sum(adj, axis=1, keepdims=True) + 1e-8)
            subg_adjs.append(adj_norm + np.eye(adj_norm.shape[0]))
        subg_adjs = np.stack(subg_adjs)

        ctx_feat = np.pad(subg_feats[:, 1:, :], ((0, 0), (1, 0), (0, 0)), 'constant')
        ctx_adj = subg_adjs

        ctr_feat = np.copy(subg_feats[:, 0:1, :])

        cls_labels = np.squeeze(self.class_labels[node_idxs])

        sample_w = (
            np.tile(np.expand_dims(cls_labels, axis=0), (n, 1)) != np.tile(np.expand_dims(cls_labels, axis=1), (1, n))
        ).astype("int32")

        sample_w = n * sample_w / np.sum(sample_w)
        sample_w += np.eye(n)

        data = {
            "ctx_adj": ctx_adj,
            "ctx_feat": ctx_feat,
            "ctr_feat": ctr_feat,
            "sample_w": sample_w,
        }

        if "ctr_idx" in batch:
            data["label"] = self.abnr_labels[batch["ctr_idx"]]

        return data

    def negetive_sampling(self, ctr_idx, clusters=None, mode="in"):
        size = self.args.neg_sampling - 1
        if clusters is not None:
            ctr_cluster = clusters[ctr_idx]
            if mode == "in":
                cands = np.where(clusters==ctr_cluster)[0]
            elif mode == "out":
                cands = np.where(clusters!=ctr_cluster)[0]
            else:
                cands = np.delete(np.arange(self.n_nodes), ctr_idx)

            cands = cands[np.where(cands!=ctr_idx)]
        else:
            cands = np.delete(np.arange(self.n_nodes), ctr_idx)

        if len(cands) >= size:
            idx = np.random.choice(cands, size=size, replace=False)
        else:
            idx = np.random.choice(cands, size=size, replace=True)
        return [ctr_idx] + idx.tolist()

    def test_batchs(self, clusters=None, mode="none"):
        batchs = []
        all_idx = np.arange(self.n_nodes)
        pbar = tqdm(all_idx)
        pbar.set_description("Sampling.")
        for idx in pbar:
            node_idxs = self.negetive_sampling(idx, clusters, mode)

            batch = {
                "node_idxs": node_idxs,
                "ctr_idx": idx
            }
            batchs.append(batch)
        return batchs

    def train_batchs(self, dynamic_batch=True):
        batchs = []
        all_idx = np.arange(self.n_nodes)
        for start_idx in range(0, self.n_nodes, self.args.batch_size):
            end_idx = start_idx + self.args.batch_size
            if end_idx > self.n_nodes:
                if not dynamic_batch:
                    break
                else:
                    end_idx = self.n_nodes
            node_idxs = all_idx[start_idx:end_idx]
            batch = {
                "node_idxs": node_idxs
            }
            batchs.append(batch)
        return batchs