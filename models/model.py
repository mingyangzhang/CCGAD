import tensorflow as tf
from layers.layers import *
from sklearn.cluster import KMeans

class Cluster(object):

    def __init__(self, k):
        self.k = k
        self.kmeans = KMeans(n_clusters=k, random_state=0)

    def __call__(self, x):
        self.kmeans.fit(x)
        return self.kmeans.labels_


class ClusterLoss(tf.keras.Model):
    def __init__(self, k):
        super(ClusterLoss, self).__init__()
        self.k = k
        self.simi = tf.keras.losses.CosineSimilarity(
            reduction=tf.keras.losses.Reduction.NONE
        )

    def call(self, inputs):
        """ c_0, c_1: shape = (N, k) """

        c_0, c_1 = inputs
        p_0 = tf.reduce_sum(c_0, axis=0)
        p_0 = tf.divide(p_0, tf.reduce_sum(p_0))

        p_1 = tf.reduce_sum(c_1, axis=0)
        p_1 = tf.divide(p_1, tf.reduce_sum(p_1))

        entrpy_0 = tf.reduce_sum(p_0 * tf.math.log(p_0)) + tf.math.log(self.k/1.0)
        entrpy_1 = tf.reduce_sum(p_1 * tf.math.log(p_1)) + tf.math.log(self.k/1.0)

        entrpy_loss = entrpy_0 + entrpy_1

        c0_norm = tf.divide(c_0, tf.reduce_sum(c_0, axis=0, keepdims=True) + 1e-8)
        c1_norm = tf.divide(c_1, tf.reduce_sum(c_1, axis=0, keepdims=True) + 1e-8)

        c0_norm = tf.transpose(c0_norm, [1, 0])
        c1_norm = tf.transpose(c1_norm, [1, 0])

        c0_rpt = tf.tile(tf.expand_dims(c0_norm, axis=1), (1, self.k, 1))
        c1_rpt = tf.tile(tf.expand_dims(c1_norm, axis=0), (self.k, 1, 1))

        c_smi =  - self.simi(c0_rpt, c1_rpt)

        diag = tf.linalg.diag_part(tf.nn.softmax(c_smi, axis=-1))
        cls_loss = - tf.reduce_mean(tf.math.log(diag + 1e-8))

        return cls_loss, entrpy_loss


class Discriminator(tf.keras.Model):
    def __init__(self, n):
        super(Discriminator, self).__init__()
        self.f_k = Bilinear(1)
        self.n = n

    def call(self, z_1, z_2):
        n = z_1.get_shape().as_list()[0]
        z1_rpt = tf.tile(tf.expand_dims(z_1, axis=1), (1, n, 1))
        z2_rpt = tf.tile(tf.expand_dims(z_2, axis=0), (n, 1, 1))
        logits = self.f_k((z1_rpt, z2_rpt))
        return tf.nn.sigmoid(logits)

class Model(tf.keras.Model):
    def __init__(self, n, n_h, k, lmbd, alpha):
        super(Model, self).__init__()

        self.mlp = MLP(1, n_h, n_h)
        self.gnn = GCNConv(self.mlp)

        self.cls = tf.keras.Sequential()
        self.cls.add(tf.keras.layers.Dense(k))
        self.cls.add(tf.keras.layers.Softmax())

        self.cls1 = tf.keras.Sequential()
        self.cls1.add(tf.keras.layers.Dense(k))
        self.cls1.add(tf.keras.layers.Softmax())

        self.n = n
        self.disc = Discriminator(n)
        self.bce = tf.keras.losses.BinaryCrossentropy()

        self.cls_loss = ClusterLoss(k)
        self.lmbd = lmbd
        self.alpha = alpha

    def call(self, inputs, training=True):
        x0, adj1, x1, sample_weight = inputs
        n = x1.get_shape().as_list()[0]

        adj0 = tf.ones((n, 1, 1))
        h_0 = self.gnn((x0, adj0), training)
        h_1 = self.gnn((x1, adj1), training)

        c_1 = self.cls1(h_1)
        c_0 = self.cls(h_0)

        cls_loss, entrpy_loss = self.cls_loss((c_0, c_1))

        prob = self.disc(h_0, h_1)[Ellipsis, 0]

        self.truth = tf.eye(n)

        prob_flat = tf.reshape(prob, [-1, 1])
        label_flat = tf.reshape(self.truth, [-1, 1])
        sample_weight = tf.reshape(sample_weight, [-1])
        cts_loss = self.bce(label_flat, prob_flat, sample_weight=sample_weight)

        loss = cts_loss + self.alpha*(cls_loss + self.lmbd*entrpy_loss)

        outs = {
            "prob": prob,
            "h0": h_0,
            "h1": h_1,
            "c0": c_0,
            "c1": c_1,
            "scalars": {
                "cts_loss": cts_loss,
                "cls_loss": cls_loss,
                "entrpy_loss": entrpy_loss
            }
        }

        return loss, outs
