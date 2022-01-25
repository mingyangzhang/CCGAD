from numpy.core.fromnumeric import resize
import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import rsqrt_grad


class SumReadout(tf.keras.layers.Layer):
    def __init__(self):
        super(SumReadout, self).__init__(self)

    def call(self, x):
        return tf.reduce_sum(x, axis=1)


class MLP(tf.keras.layers.Layer):
    """MLP"""

    def __init__(self, num_layers, hidden_dim, output_dim, linear=True):

        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.linear = linear

    def build(self, input_shape):
        self.mlp = tf.keras.Sequential()
        for _ in range(self.num_layers - 1):
            self.mlp.add(tf.keras.layers.Dense(self.hidden_dim))
            self.mlp.add(tf.keras.layers.BatchNormalization())
            self.mlp.add(tf.keras.layers.ReLU())
        if self.linear:
            self.mlp.add(tf.keras.layers.Dense(self.output_dim, use_bias=False))
        else:
            self.mlp.add(tf.keras.layers.Dense(self.output_dim, use_bias=True))
            # self.mlp.add(tf.keras.layers.PReLU())

    def call(self, x, training):
        return self.mlp(x, training)


class Bilinear(tf.keras.layers.Layer):
    """ Bilinear layer. """

    def __init__(self, out_dim):
        super(Bilinear, self).__init__()
        self.out_dim = out_dim

    def build(self, input_shape):
        in_dim1 = input_shape[0][-1]
        in_dim2 = input_shape[1][-1]
        self.w = self.add_weight(name="w",
                                 shape=(self.out_dim, in_dim1, in_dim2),
                                 initializer="random_normal")
        self.b = self.add_weight(name="b",
                                 shape=(self.out_dim,),
                                 initializer="random_normal")

    def call(self, inputs):
        x1, x2 = inputs
        y = tf.einsum('onm,ijm->ijon', self.w, x2)
        out = tf.einsum('ijon,ijn->ijo', y, x1) + self.b
        return out


class GCNConv(tf.keras.layers.Layer):
    """ Graph Convolutional Network. """

    def __init__(self, apply_func):
        super(GCNConv, self).__init__()
        self.apply_func = apply_func

    def build(self, input_shape):
        self.act = tf.keras.layers.PReLU()
        self.bias = self.add_weight(name="bias", shape=(1,), initializer="zeros")
        self.dropout = tf.keras.layers.Dropout(0.0)

    def call(self, inputs, training=True):
        seq_fts, supports = inputs
        if self.apply_func is not None:
            seq_fts = self.apply_func(seq_fts, training)
        rst = self.act(tf.matmul(supports, seq_fts) + self.bias)
        rst = self.dropout(rst, training)
        return tf.reduce_mean(rst, axis=1)


class GINConv(tf.keras.layers.Layer):
    """ Graph Isomorphism Network. """

    def __init__(self, apply_func, init_eps=0, learn_eps=False):
        super(GINConv, self).__init__()
        self.apply_func = apply_func
        self.init_eps = init_eps
        self.learn_eps = learn_eps

    def build(self, input_shape):
        self.eps = self.add_weight(name="eps",
                                   initializer=tf.constant_initializer(self.init_eps),
                                   trainable=self.learn_eps)

    def call(self, inputs, training=True):
        seq_fts, supports = inputs
        # neighbor_agg = tf.matmul(supports, seq_fts)
        rst = (1 + self.eps) * seq_fts
        if self.apply_func is not None:
            rst = self.apply_func(rst, training)
        return tf.reduce_mean(rst, axis=1)


class GIN(tf.keras.layers.Layer):
    def __init__(self, h_dim, out_dim, n_layers, n_mlp_layers, learn_eps, graph_pooling_type, dropout):
        super(GIN, self).__init__()
        self.n_layers = n_layers
        self.learn_eps = learn_eps
        self.n_mlp_layers = n_mlp_layers
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.graph_pooling_type = graph_pooling_type
        self.learn_eps = learn_eps
        self.dropout = dropout

    def build(self, input_shape):

        # List of MLPs
        self.ginlayers = []
        self.batch_norms = []

        for _ in range(self.n_layers - 1):
            mlp = MLP(self.n_mlp_layers, self.h_dim, self.h_dim)
            self.ginlayers.append(GINConv(mlp, 0, self.learn_eps))
            self.batch_norms.append(tf.keras.layers.BatchNormalization())

        self.linears_prediction = []
        for _ in range(self.n_layers):
            lp = tf.keras.Sequential()
            lp.add(tf.keras.layers.Dense(self.out_dim))
            lp.add(tf.keras.layers.PReLU())
            self.linears_prediction.append(lp)

        self.dropout = tf.keras.layers.Dropout(self.dropout)

        if self.graph_pooling_type == "sum":
            self.pool = SumReadout()
        else:
            raise NotImplementedError

    def call(self, inputs, training=True):
        h, adj = inputs
        # list of hidden representation at each layer (including input)
        hidden_rep = [h]

        for i in range(self.n_layers - 1):
            h = self.ginlayers[i](h, adj, training)
            h = self.batch_norms[i](h, training)
            h = tf.nn.relu(h)
            hidden_rep.append(h)

        score_over_layer = 0

        # perform pooling over all nodes in each graph in every layer
        all_outputs = []
        for i, h in list(enumerate(hidden_rep)):
            pooled_h = self.pool(h)
            all_outputs.append(pooled_h)
            score_over_layer += self.dropout(
                self.linears_prediction[i](pooled_h), training
            )

        return score_over_layer
