"""Try and train some chap on some kind of data perhaps"""
import os

import tensorflow as tf

from hrr import SimpleHRRCell
import read_occult as data


flags = tf.app.flags
flags.DEFINE_integer('embedding_size', 32, 'size of char embeddings')
flags.DEFINE_integer('width', 128, 'size of hidden state')
flags.DEFINE_integer('batch_size', 32, 'number to do at once')
flags.DEFINE_integer('sequence_length', 100, 'how much to truncate bptt')
flags.DEFINE_integer('replicas', 1, 'how many copies of mem')
flags.DEFINE_float('learning_rate', 0.001, 'step size for ADAM')

FLAGS = flags.FLAGS


def forward(inputs, vocab_size):
    """Gets the forward pass of the model, not including sampling.
    Most of the parameters are taken from FLAGS.

    Args:
        inputs: a big variable representing the input batches.
        vocab_size: the number of input symbols.

    Returns:
        logits, final_state, initial_state: the raw output of the net,
            the state at the end and the variables representing the initial
            state.
    """

    with tf.variable_scope('embedding'), tf.device('/cpu:0'):
        embedding = tf.get_variable('embedding',
                                    [vocab_size, FLAGS.embedding_size])
        inputs = tf.unpack(tf.nn.embedding_lookup(embedding, inputs), axis=1)

    cell = SimpleHRRCell(FLAGS.width, FLAGS.replicas)

    initial_state = cell.zero_state(FLAGS.batch_size, tf.float32)

    outputs, final_state = tf.nn.rnn(cell, inputs, initial_state=initial_state)

    with tf.variable_scope('output_layer'):
        # softmax it, faster to do it all at once by catting the sequence
        outputs = tf.reshape(tf.concat(1, outputs), [-1, FLAGS.width*2])
        # put a softmax on top
        logits = tf.nn.bias_add(
            tf.matmul(
                outputs, tf.get_variable("softmax_W",
                                         [FLAGS.width*2, vocab_size])),
            tf.get_variable("softmax_b", [vocab_size]))

    return logits, final_state, initial_state


def main(_):

    print('{:~^50}'.format('getting data'), end='', flush=True)
    # get the data
    train, valid, test, vocab = data.occult_raw_data('occulttraining.txt')
    inv_vocab = {b: a for a, b in vocab.items()}
    with tf.variable_scope('data'):
        inputs = tf.placeholder(tf.int32,
                                [FLAGS.batch_size, FLAGS.sequence_length],
                                'inputs')
        targets = tf.placeholder(tf.int32,
                                 [FLAGS.batch_size, FLAGS.sequence_length],
                                 'targets')
    print('\r{:-^50}'.format('got data'))

    print('{:~^50}'.format('getting forward'), end='', flush=True)
    with tf.variable_scope('rnn_model'):
        logits, final_state, initial_state = forward(inputs, len(vocab))
        # and get some tensors to sample a sequence

    print('\r{:-^50}'.format('got forward model'))
    print('{:~^50}'.format('getting training ops'), end='', flush=True)
    with tf.variable_scope('training'):
        loss_op = tf.nn.seq2seq.sequence_loss_by_example(
            [logits], [tf.reshape(targets, [-1])],
            [tf.ones([FLAGS.batch_size * FLAGS.sequence_length])], len(vocab))
        loss_op = tf.reduce_mean(loss_op)

        opt = tf.train.AdamOptimizer(FLAGS.learning_rate)

        train_op = opt.minimize(loss_op)
    print('\r{:-^50}'.format('got train ops'))

    sess = tf.Session()
    print('{:~^50}'.format('initialising'), end='', flush=True)
    sess.run(tf.initialize_all_variables())
    print('\r{:-^50}'.format('initialised'))

    # move this to a function (and, you know, do it more than once)
    # also don't forget to pass the initial state along
    for x, y in data.occ_iterator(train,
                                  FLAGS.batch_size,
                                  FLAGS.sequence_length):
        loss, _ = sess.run([loss_op, train_op],
                           {inputs: x, targets: y})
        print('\rloss: {}'.format(loss), end='', flush=True)
    print()


if __name__ == '__main__':
    tf.app.run()
