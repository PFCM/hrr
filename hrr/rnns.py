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
        more complex (loljk), but we will see.

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
        return (self._width*2, self._width*self._replicas*2)

    @property
    def output_size(self):
        return self._width*2

    def zero_state(self, batch_size, dtype):
        """Gets an initial state (and permutations?)
        """
        initial_out = tf.zeros([batch_size, self._width*2], dtype=dtype)
        mem = hrr.get_holographic_memory(self._width, batch_size,
                                         self._replicas, dtype=dtype)
        return (initial_out, mem)

    def __call__(self, inputs, state, scope=None):
        """Does the stuff"""
        with tf.variable_scope(scope or type(self).__name__):
            h, mem = state

            concat = _linear(tf.concat(1, [h, inputs]),
                             4*self._width, True, scope='keys')
            concat = tf.nn.tanh(concat)
            in_key, out_key = tf.split(1, 2, concat)

            in_val = _linear(inputs, self._width*2, True, scope='input')
            in_val = in_val

            updated_mem = hrr.store(in_key, in_val, mem)
            output = hrr.retrieve(out_key, updated_mem)

            output = tf.nn.tanh(output)

            return output, (output, updated_mem)
