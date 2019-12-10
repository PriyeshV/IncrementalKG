import tensorflow as tf
from src.layers.layer import Layer
from src.utils.inits import glorot, const, tanh_init
from src.utils.utils import dot


class Kernels_new(Layer):

    def __init__(self, layer_id, x_names, adj_name, dims, dropout, act=tf.nn.relu, sparse_inputs=False,
                 bias=False, shared_weights=True, skip_connection=True, weights=True, **kwargs):
        super(Kernels_new, self).__init__(**kwargs)

        self.layer_id = layer_id
        self.act = act
        self.dropout = dropout
        self.skip_connetion = skip_connection
        self.sparse_inputs = sparse_inputs

        self.weights = weights
        self.bias = bias
        self.shared_weights = shared_weights
        self.node = None
        self.neighbor = None
        self.g0 = None
        self.g1 = None

        self.adj_name = adj_name
        self.node_feautures = x_names[0]
        self.neighbor_features = x_names[1]

        # Weights initialization
        self.output_dim = dims[layer_id+1]

        self.node_dims = 0
        self.neigh_dims = 0

        if self.bias:
            self.vars['bias'] = const([self.output_dim])

        self.weights_node = {}
        self.bias_node = None
        self.weights_neigh = {}
        self.bias_neigh = None

        if weights:
            with tf.compat.v1.variable_scope(self.name + "_neighbor_vars_out"):
                self.weights_neigh['out'] = tanh_init((self.output_dim, self.output_dim), name='weights_out')
            with tf.compat.v1.variable_scope(self.name + "_neighbor_vars_in"):
                self.weights_neigh['in'] = tanh_init((self.output_dim, self.output_dim), name='weights_in')


    def compute_features(self, inputs, weights, bias, keys, n_nodes):
            if len(keys) == 0:
                return tf.zeros(shape=(tf.cast(n_nodes, dtype=tf.int32), self.output_dim))
            output = tf.zeros(shape=(self.output_dim), dtype=tf.float32)
            for key in keys:
                data = inputs[key]

                dropout = self.dropout
                data = tf.nn.dropout(data, 1 - dropout)
                output += dot(data, weights[key], sparse=False)
            return output

    def combine(self):
        node = tf.multiply(tf.expand_dims(self.g0, axis=1), self.node)
        neighbor = tf.multiply(tf.expand_dims(self.g1, axis=1), self.neighbor)
        return node + neighbor


class Kernels(Layer):

    def __init__(self, **kwargs):
        super(Kernels, self).__init__(**kwargs)

    def combine(self, g0, g1, node, neighbors, node_W, neigh_W):
        node = tf.matmul(tf.multiply(tf.expand_dims(g0, axis=0), node), node_W)
        neighbors = tf.matmul(tf.multiply(tf.expand_dims(g1, axis=0), neighbors), neigh_W)
        return node + neighbors