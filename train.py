"""Try and train some chap on some kind of data perhaps"""
import os
import datetime
import shutil
import collections

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
flags.DEFINE_integer('num_epochs', 10, 'how long to train')
flags.DEFINE_string('logdir', '/tmp/tflogs', 'where to store checkpoints etc.')
flags.DEFINE_bool('fresh_start', True, 'whether to start fresh or not')

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


# TODO finish these two (to handle replicated mem)
def _eval_state(state_tuple):
    """Evaluates an HRR cell state tuple"""
    return [var.eval() for var in state_tuple]


def _pack_state_dict(state_var, state_vals):
    """Appropriately packs a state into a feed dict"""
    return {var: val for var, val in zip(state_var, state_vals)}


def run_epoch(dataset, eval_op, train_op, input_var, target_var, istatevar,
              fstatevar):
    """Runs an epoch of training"""
    # get numpy values for the initial state to pass along
    state = _eval_state(istatevar)

    total_loss = 0
    total_steps = 1
    sess = tf.get_default_session()
    # TODO deal with replicas
    fstatevar = list(fstatevar)
    for x, y in data.occ_iterator(dataset,
                                  FLAGS.batch_size,
                                  FLAGS.sequence_length):
        feed = _pack_state_dict(istatevar, state)
        feed[input_var] = x
        feed[target_var] = y
        results = sess.run([eval_op, train_op] + fstatevar,
                           feed)

        loss = results[0]

        state = results[2:]
        # TODO deal with replicas here too
        # state = state[0]

        print('\rloss: {}'.format(loss), end='', flush=True)
        total_loss += loss
        total_steps += 1
    print()
    return total_loss / total_steps


def main(_):
    # make sure we're not doubling up
    if FLAGS.logdir == '/tmp/tflogs':
        FLAGS.logdir = os.path.join(
            FLAGS.logdir, datetime.datetime.now().strftime('%Y-%m-%dT%H%M:%S'))
    elif os.path.exists(FLAGS.logdir) and FLAGS.fresh_start:
        print('(clearing out {})'.format(FLAGS.logdir))
        shutil.rmtree(FLAGS.logdir)

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
    noop = tf.no_op()
    sv = tf.train.Supervisor(logdir=FLAGS.logdir, summary_op=None)

    print('{:~^50}'.format('initialising (preparing session)'),
          end='', flush=True)
    with sv.managed_session() as sess:
        print('\r{:-^50}'.format('ready'))

        with sess.as_default():
            for epoch in range(FLAGS.num_epochs):
                train_loss = run_epoch(train, loss_op, train_op, inputs,
                                       targets, initial_state, final_state)
                valid_loss = run_epoch(valid, loss_op, noop, inputs,
                                       targets, initial_state, final_state)

                print('~Epoch {}'.format(epoch+1))
                print('~~train loss: {}'.format(train_loss))
                print('~~valid loss: {}'.format(valid_loss))


if __name__ == '__main__':
    tf.app.run()
