"""RNNs using holographic reduced representations to store their state.
May even eventually include the Associative LSTM"""
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import _linear

import hrr


class SimpleHRRCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, width, replicas=1):
        """The simplest RNN cell I can think of that uses HRRs for its state.

        At each step, it will produce a storage key and a retrieval key.
        Both keys will be dependent on the previous output and the current
        input. I would much rather be able to use the hidden state rather
        than the previous output, but I can't figure out how to make that
        play nice with the replicas.

        Given the additive nature of the HRR composition, this will probably
        explode. Therefore, there is a good chance we will need to do something
        more complex, but we will see.

        Args:
            width (int): the size of the memory. The actual state vectors will
                be twice the size of this because they are actually complex
                numbers.
            replicas (Optional[int]): how many copies of the memory to use.
        """
        self._width = width
        self._replicas = replicas

    @property
    def state_size(self):
        return (self._width, self._width*self._replicas)

    @property
    def output_size(self):
        return self._width

    def zero_state(self, batch_size, dtype):
        """Gets an initial state (and permutations?)
        """
        pass

    def __call__(self, inputs, state, scope=None):
        """Does the stuff"""
        with tf.variable_scope(scope or type(self).__name__):
            h, mem = state
