from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import time

import tensorflow.python.platform

import numpy as np
from six.moves import xrange
import tensorflow as tf

from tensorflow.python.platform import gfile


def _read_chars(filename):
    with gfile.GFile(filename, "r") as f:
        return list(f.read())


def _build_vocab(filename):
    data = _read_chars(filename)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])

    words, other = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def _file_to_char_ids(filename, word_to_id):
    data = _read_chars(filename)
    return [word_to_id[word] for word in data]


def labels_to_one_hot(l, num):
    num_labels = l.shape[0]
    index_offset = np.arange(num_labels) * num
    labels_one_hot = np.zeros((num_labels, num))
    labels_one_hot.flat[index_offset + l.ravel()] = 1
    return labels_one_hot


def occult_raw_data(path=None, one_hot=True):
    """Loads the occult file.
    Returns:
      tuple (train_data, valid_data, test_data, vocab)
      each can be passed to the iterator function (except vocab)
    """
    word_to_id = _build_vocab(path)
    all_data = _file_to_char_ids(path, word_to_id)
    # it's just one massive sequence!
    # now we need to slice it up
    # into train, test and valid
    num_chars = len(all_data)
    all_data = np.array(all_data)
    all_onehot = all_data
    train_end = num_chars//100 * 98
    train = all_onehot[:train_end, ...]
    test_end = train_end + num_chars//100
    test = all_onehot[train_end:test_end, ...]
    valid = all_onehot[test_end:, ...]

    return train, valid, test, word_to_id


def occ_iterator(raw_data, batch_size, num_steps):
    """Iterate the data.
    Generates batch_size pairs.

    Yields:
        pairs of batched data, the second element is the first shifted to
        the right by one
    """
    raw_data = np.array(raw_data, dtype=np.int32) # jic

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size): # unconvincing
        data[i] = raw_data[batch_len * i:batch_len * (i+1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (x,y)
