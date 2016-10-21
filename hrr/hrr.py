"""Some things for handling holographic reduced representations"""
import collections

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
    perm = tf.concat(0, [perm, perm + length])

    return tf.get_variable(name, initializer=perm)


def _make_trace(key, value, key_split=1, val_split=1):
    """combine two (batches of) items to make a memory trace.

    Args:
        key: the first item
        value: the second
        key_split: the axis to split the key on.
        val_split: the axis to split the value on.

    Returns:
        the combined trace (by element-wise complex multiplication)
    """
    # pull out the real and imaginary parts
    re_k, im_k = tf.split(key_split, 2, key)
    re_v, im_v = tf.split(val_split, 2, value)
    # if we had to do the split on different axis, we need to make sure the
    # shapes line up for broadcasting
    if key_split > val_split:
        re_v = tf.expand_dims(re_v, key_split-1)
        im_v = tf.expand_dims(im_v, key_split-1)
    elif val_split > key_split:
        re_k = tf.expand_dims(re_k, val_split-1)
        im_k = tf.expand_dims(im_k, val_split-1)

    # element-wise complex multiply
    re_t = (re_k * re_v) - (im_k * im_v)
    im_t = (re_k * im_v) + (im_k * re_v)

    # stick them back together
    # might need squeezing with a single one
    # also this doesn't properly respect weird split dimensions
    return tf.concat(len(re_t.get_shape())-1, [re_t, im_t])


def _apply_perms(key, perms):
    """Applies permutations to a batch of keys.

    Args:
        key: [batch_size, width*2] tensor of keys to be permuted
        perms: list of [width*2] tensors of ints with permutations

    Returns:
        [batch_size, len(perms), width*2] tensor of permuted keys
    """
    # tf.gather works on the first index
    # so the easiest thing to do is transpose the keys before and after
    key_t = tf.transpose(key)
    permed_keys = [tf.transpose(tf.gather(key_t, perm))
                   for perm in perms]
    permed_keys = tf.pack(permed_keys, 1)

    return permed_keys


def get_holographic_memory(width, batch_size, replicas=1, scope=None):
    """Pretty much just gets a big tensor. We work a Fourier transform away
    from Plate's initial circular convolution based presentation, which means
    that whenever we deal with things that operated on the representation,
    everything is twice as large as it seems because we store real and
    imaginary parts separately.

    Args:
        width (int): the size of the vectors involved, in terms of complex
            numbers.
        batch_size (int): how many to do at once.
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
                           initializer=tf.constant_initializer(0.0)):
        rep = tf.get_variable('hrr', shape=[batch_size, replicas, width * 2])

        if replicas > 1:
            indices = tf.range(width)
            perms = [_make_perm(indices, width, 'perm_{}'.format(i))
                     for i in range(replicas)]

            return [rep] + perms

    return rep


def store(key, value, memory):
    """Makes an appropriate trace out of (key, value) and updates
    the memory (which is to say, adds ops to do these things).

    args:
        key (tensor): float tensor with real part followed by imaginary.
        value (tensor): value tensor, again real followed by imaginary.
        memory: whatever is returned from `get_holographic_memory`.

    returns:
        updated memory tensor.
    """
    # what are we dealing with
    if not isinstance(memory, collections.Sequence):
        trace = _make_trace(key, value)
        trace = tf.expand_dims(trace, 1)
        return memory + trace
    else:
        # get all the permuted keys
        permed_keys = _apply_perms(key, memory[1:])
        # now keys are appropriately shaped for the memory replicas
        # make their traces with value
        traces = _make_trace(permed_keys, value, key_split=2)
        return [memory[0] + traces] + memory[1:]


def retrieve(inv_key, memory):
    """Retrieves a batch of keys from a batch of replicas of memories.

    Args:
        inv_key (tensor): the inverse key to use for the lookup.
        memory (tensor): the result of store.

    Returns:
        the item retrieved from the memory.
    """
    # fingers crossed for some broadcasting
    if not isinstance(memory, collections.Sequence):
        return tf.squeeze(_make_trace(inv_key, memory, val_split=2))
    else:
        # have to apply all perms, retrieve the lot and average
        permed_keys = _apply_perms(inv_key, memory[1:])
        # now the retrieval
        replicas = _make_trace(permed_keys, memory[0],
                               key_split=2, val_split=2)
        return tf.reduce_mean(replicas, reduction_indices=[1])


if __name__ == '__main__':
    # run a quick test to make sure things make some sense
    width = 128
    batch_size = 32
    replicas = 1
    mem = get_holographic_memory(width, batch_size, replicas=replicas)

    key = tf.get_variable('key', shape=[batch_size, width*2])
    value = tf.get_variable('val', shape=[batch_size, width*2])
    # invert the key by taking its complex conjugate
    re_k, im_k = tf.split(1, 2, key)
    inv_key = tf.concat(1, [re_k, -im_k])

    updated_mem = store(key, value, mem)
    mem = updated_mem if replicas == 1 else updated_mem[0]

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    print('key')
    print(key.eval())
    print('value')
    print(value.eval())

    print('memory (all replicas)')
    print(mem.eval())

    retrieved = retrieve(inv_key, updated_mem)
    print('retrieved')
    print(retrieved.eval())

    print('difference')
    print((value - retrieved).eval())

    print('mean squared error')
    print(tf.reduce_mean(tf.squared_difference(value, retrieved)).eval())
