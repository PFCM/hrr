"""Some things for handling holographic reduced representations"""
import tensorflow as tf


def _make_perm(indices, length, name):
    """Makes a permutation we can use with tf.gather. Repeats the permutation
    twice with an offset so that the it is applied appropriately to both the
    real and the imaginary parts.

    Args:
        indices: a range containing all the indices we need.
        length: how many indices there are.
        name: a name for the operations.

    Returns:
        an int32 tensor of indices.
    """
    perm = tf.random_shuffle(indices)
    perm = tf.concat(1, [perm, perm + length])

    return tf.get_variable(name, initializer=perm)


def get_holographic_memory(width, replicas=1, scope=None):
    """Pretty much just gets a big tensor. We work a Fourier transform away
    from Plate's initial circular convolution based presentation, which means
    that whenever we deal with things that operated on the representation,
    everything is twice as large as it seems because we store real and
    imaginary parts separately.

    Args:
        width (int): the size of the vectors involved, in terms of complex
            numbers.
        replicas (Optional[int]): how many copies the memory has.
        scope (Optional): a variable scope for the added operations.

    Returns:
        tensor: a blank representation, which is just a float tensor with shape
            `[replicas, width * 2]`, with a zero initialiser. If replicas > 1,
            then we actually return a sequence where the first element is the
            tensor as described and the remaining `replicas` items are the
            permutations we need to decorrelate the replicas.
    """
    with tf.variable_scope(scope or 'hrr',
                           initialiser=tf.constant_initializer(0.0)):
        rep = tf.get_variable('hrr', size=[replicas, width * 2])

        if replicas > 1:
            indices = tf.range(width)
            perms = [_make_perm(indices, width, 'perm_{}'.format(i))
                     for i in range(replicas)]

        return [rep] + perms

    return rep


def store(key, value, memory):
    """Makes an appropriate trace out of (key, value) and appropriately updates
    the memory.

    args:
        key (tensor): float tensor with real part followed by imaginary.
        value (tensor): value tensor, again real followed by imaginary.
        memory: whatever is returned from `get_holographic_memory`.

    returns:
        updated memory tensor.
    """
    pass
