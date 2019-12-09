import tensorflow as tf


def get_tf_unnormalize_adj(adjmat, degrees):
    degrees = tf.expand_dims(1 / degrees, axis=1)
    unnorm_adjmat = adjmat.__mul__(degrees)
    return unnorm_adjmat


def get_tf_normalize_adj(adjmat, degrees):
    degrees = tf.expand_dims(1/tf.sqrt(degrees), axis=1)
    laplacian = adjmat.__mul__(degrees)
    degrees = tf.transpose(degrees, [1, 0])
    norm_adjmat = laplacian.__mul__(degrees)
    return norm_adjmat


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse.sparse_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res