from src.utils.utils import *
from src.layers.graph_convolutions.kernel import Kernels_new


class Kernel(Kernels_new):

    def __init__(self, **kwargs):
        super(Kernel, self).__init__(**kwargs)

    def _call(self, data):
        input_h = data['activations'][-1]
        h_out = tf.matmul(tf.sparse.sparse_dense_matmul(data['adj_out_mat'], input_h) + data['h_rel_out'], self.weights_neigh['out'])
        h_in = tf.matmul(tf.sparse.sparse_dense_matmul(data['adj_in_mat'], input_h) + data['h_rel_in'], self.weights_neigh['in'])

        h = h_out + h_in
        if self.bias:
            h += self.vars['bias']
        return h