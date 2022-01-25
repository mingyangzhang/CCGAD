import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['DGLBACKEND'] = 'tensorflow'

import pickle
import argparse
import datetime
import numpy as np
import pandas as pd
np.random.seed(0)

import tensorflow as tf
import tensorflow_addons as tfa

tf.random.set_seed(0)

import json
import scipy.io as sio
from tqdm import tqdm
from collections import Counter

from data_loaders.graph_loader import GraphLoader
from models.model import Model
from sklearn.metrics import roc_auc_score
from utils.utils import precision_recall_k

import sys
import signal
signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))


parser = argparse.ArgumentParser(description='Contrastive Graph Anomaly Detection')
parser.add_argument('--graph_path', type=str, default='./dataset/')

parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--neg_sampling', type=int, default=0)

parser.add_argument('--subgraph_size', type=int, default=5)
parser.add_argument('--n_cluster', type=int, default=10)
parser.add_argument('--lmbd', type=float, default=0.01)
parser.add_argument('--alpha', type=float, default=1.0)

parser.add_argument('--rw_sample', type=str, default="load")
parser.add_argument('--neg_stat', type=str, default="mean")
parser.add_argument('--mode', type=str, default='out')
parser.add_argument('--train', type=str, default="true")
parser.add_argument('--load', type=str, default="false")
parser.add_argument('--out_dir', type=str, default="outputs")

parser.add_argument('--flag', type=str, default="")

args = parser.parse_args()

if args.neg_sampling == 0:
    args.neg_sampling = args.batch_size

print(args)

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)


data_loader = GraphLoader(args)
model = Model(args.batch_size, args.embedding_dim, args.n_cluster, args.lmbd, args.alpha)

lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=args.lr, decay_steps=1000, decay_rate=0.9)
optimizer = tfa.optimizers.AdamW(learning_rate=lr_scheduler, beta_1=0.9, clipvalue=1, weight_decay=args.weight_decay)


# tensorboard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)


@tf.function
def train_step(inputs):
    with tf.GradientTape() as tape:
        loss, outs = model(inputs, True)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, outs["scalars"]

def train_model(train_data, test_data):
    min_loss = np.inf
    pbar = tqdm(range(args.num_epoch))
    for epoch in pbar:
        if epoch == 0:
            pbar.set_description(f"Epoch {epoch} loss: nan")
        else:
            pbar.set_description(f"Epoch {epoch} loss: {loss:.4f}")
        loss_list = []
        scalar_list = []
        for batch in train_data.train_batchs():
            data = train_data.batch_data(batch)
            # contrastive nodes
            feat0 = data['ctr_feat']
            adj1 = data['ctx_adj']
            feat1 = data['ctx_feat']
            sample_w = data['sample_w']

            loss, scalars = train_step((feat0, adj1, feat1, sample_w))
            loss_list.append(loss.numpy())
            scalars = {k:v.numpy() for k,v in scalars.items()}
            scalars.update({"lr": optimizer._decayed_lr('float32').numpy()})
            scalar_list.append(scalars)

        loss = np.mean(loss_list)
        scalars_df = pd.DataFrame(scalar_list)

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=epoch)
            for k, v in scalars_df.mean().items():
                tf.summary.scalar(k, v, step=epoch)

        if loss < min_loss:
            min_loss = loss
            model.save_weights(f'./checkpoints/{args.dataset}')

        # if  epoch > 0 and epoch % 100 == 0:
        #     test_model(test_data, args.mode)


@tf.function
def test_step(inputs):
    prob = model.disc(*inputs)
    return prob[Ellipsis, 0]


@tf.function
def cluster_step(inputs):
    _, outs = model(inputs, False)
    return outs["h0"], outs["h1"], outs["c0"], outs["c1"]


def neg_score_stats(neg_scores, stat_mode="mean"):
    if stat_mode == "mean":
        return np.mean(neg_scores)
    else:
        return np.max(neg_scores)


def cluster_nodes(test_data):
    ctx_clusters = []
    ctx_hs = []
    ctr_clusters = []
    ctr_hs = []

    pbar = tqdm(test_data.train_batchs(dynamic_batch=True))
    pbar.set_description("Clustering")
    for batch in pbar:
        data = test_data.batch_data(batch)

        feat0 = data['ctr_feat']
        adj1 = data['ctx_adj']
        feat1 = data['ctx_feat']
        sample_w = data['sample_w']

        h_0, h_1, c_0, c_1 = cluster_step((feat0, adj1, feat1, sample_w))

        ctx_hs.append(h_1.numpy())
        ctx_clusters.append(np.argmax(c_1.numpy(), axis=-1))

        ctr_hs.append(h_0.numpy())
        ctr_clusters.append(np.argmax(c_0.numpy(), axis=-1))

    ctx_clusters = np.concatenate(ctx_clusters)
    ctx_hs = np.concatenate(ctx_hs)

    ctr_clusters = np.concatenate(ctr_clusters)
    ctr_hs = np.concatenate(ctr_hs)
    return ctx_clusters, ctx_hs, ctr_clusters, ctr_hs


def test_model(test_data, mode):
    labels = []
    scores = []
    pos_probs = []
    neg_probs = []

    ctx_clusters, ctx_hs, ctr_clusters, ctr_hs = cluster_nodes(test_data)

    print(Counter(ctx_clusters))
    pbar = tqdm(test_data.test_batchs(ctx_clusters, mode))

    for batch in pbar:
        node_idxs = batch["node_idxs"]
        ctr_idx = batch["ctr_idx"]

        # center node
        h0 = ctr_hs[node_idxs]
        h1 = ctx_hs[node_idxs]
        prob = test_step((h0, h1))

        prob = prob.numpy()[0, :]
        neg_prob = neg_score_stats(prob[1:], args.neg_stat)
        pos_prob = prob[0]

        scores.append(neg_prob - pos_prob)
        labels.append(test_data.abnr_labels[ctr_idx])
        pos_probs.append(pos_prob)
        neg_probs.append(neg_prob)

    scores = np.array(scores)
    pos_probs = np.array(pos_probs)
    neg_probs = np.array(neg_probs)

    labels = np.concatenate(labels)
    auc = roc_auc_score(labels, scores)
    print(f"AUC: {auc:.4f}")

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logs = {"args": vars(args), "auc": auc}

    k_list = [10, 50, 100, 200, 300]
    for k in k_list:
        pk, rk = precision_recall_k(labels, scores, k)
        print(f"Precision@{k}: {pk:.4f}; Recall@{k}: {rk:.4f};")
        logs.update({f"P@{k}": pk, f"R@{k}": rk})

    if not os.path.exists(f"{args.out_dir}/{args.dataset}/"):
        os.mkdir(f"{args.out_dir}/{args.dataset}/")

    with open(f"{args.out_dir}/{args.dataset}/{current_time}", "w") as fw:
        json.dump(logs, fw, indent=4)

    return scores, labels, ctx_hs, ctx_clusters, ctr_hs, ctr_clusters, pos_probs, neg_probs, auc

if __name__ == "__main__":

    if args.load == "true":
        model.load_weights(f'./checkpoints/{args.dataset}')

    if args.train == "true":
        train_model(data_loader, data_loader)

    model.load_weights(f'./checkpoints/{args.dataset}')
    scores, labels, ctx_h, ctx_clusters, ctr_h, ctr_clusters, pos_probs, neg_probs, auc = test_model(data_loader, args.mode)

    states_to_save = {
        "scores": scores,
        "labels": labels,
        "pos_probs": pos_probs,
        "neg_probs": neg_probs,
        "ctr_h": ctr_h,
        "ctr_clusters": ctr_clusters,
        "ctx_h": ctx_h,
        "ctx_clusters": ctx_clusters
    }

    with open(f"outputs/{args.dataset}{args.flag}.pkl", "wb") as fw:
        pickle.dump(states_to_save, fw)
