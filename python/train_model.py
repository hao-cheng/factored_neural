#!/usr/bin/env python
"""This is the training script for the model.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys
import cPickle as pickle
import time
import numpy as np


import tensorflow as tf
from model import helper
from model.interact_seq2seq_model import InteractiveSeq2SeqModel


def config_param():
    """Creates the configuration object for training.

    Returns:
        A TrainConfig object.
    """
    config = helper.TrainConfig()
    # All those parameters should be configured accordingly.
    config.batch_size = 1
    config.vocab_size = 128
    config.embed_dim = 10
    config.hidden_dim = 10
    config.num_hiddens = 1
    config.mode_dim = 10
    config.num_modes = 10
    config.cmt_seq_len = 10
    config.reply_seq_len = 5
    config.alpha = 0.1
    config.initializer = 'Gaussian'
    config.stddev = 0.1
    config.opt_method = 'Adam'
    config.learning_rate = 1e-3
    config.keep_prob = 1.0
    config.forget_bias = 1.0
    config.lr_decay = 0.5
    config.max_epoch_iter = 3
    # Special reserved token indices.
    config.EOS_ID = 2
    config.PAD_ID = 0
    config.UNK_ID = 1
    config.GO_ID = 3
    return config


def run_epoch(model, session, tuple_data, eval_op, PAD_ID=0, verbose=True):
    """Runs an epoch over the data.

    Args:
        model: Model object.
        session: tf.Session object.
        tuple_data: Tuple of comment, reply data.
        eval_op: Tensorflow operation to be carried out.
        PAD_ID: Integer for the padding symbol.
        verbose: Bool whether to be verbose.

    Returns:
        total_loss: Accumulated loss for this epoch.
    """
    start_time = time.time()
    total_loss = 0.0
    for it in xrange(len(tuple_data) // model.batch_size + 1):
        c_ins, r_ins, c_wts, r_wts, c_lens, r_lens = helper.get_batch_data(
            tuple_data,
            model.cmt_seq_len,
            model.reply_seq_len,
            model.batch_size,
            PAD_ID=PAD_ID
        )

        feed_dict = {}
        for l in xrange(model.cmt_seq_len):
            feed_dict[model.comment_inputs[l]] = c_ins[l]
            feed_dict[model.comment_weights[l]] = c_wts[l]
            # feed_dict[model.comment_last_inputs_weights[l]] = c_last_ins_ws[l]
        for l in xrange(model.reply_seq_len):
            feed_dict[model.reply_inputs[l]] = r_ins[l]
            feed_dict[model.reply_weights[l]] = r_wts[l]

        fetches = [model.total_loss, eval_op]
        loss, _ = session.run(fetches, feed_dict)
        total_loss += loss
        if verbose and (it + 1) % 100 == 0:
            print('iter {:d}: loss {:.3f}, {:.3f} example per seconds'.format(
                it,
                total_loss / it,
                it * model.batch_size / (time.time() - start_time)))

    print('time for one epoch: {:.3f} secs'.format(time.time() - start_time))
    return total_loss


def train_model(train_data, valid_data, train_config, word_embed=None):
    tf.set_random_seed(0)
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=sess_config) as session:
        if train_config.initializer == 'Uniform':
            initializer = tf.random_uniform_initializer(
                -train_config.scale, train_config.scale, dtype=tf.float32)
        elif train_config.initializer == 'Gaussian':
            initializer = tf.truncated_normal_initializer(
                mean=0, stddev=train_config.stddev, dtype=tf.float32)
        else:
            print('Unknown initilaizer {0}'.format(train_config.initializer),
                  file=sys.stderr)
            sys.exit(1)

        with tf.variable_scope('model', reuse=None, initializer=initializer):
            model = InteractiveSeq2SeqModel(train_config, mode='TRAIN',
                                            loaded_word_embed=word_embed)
        with tf.variable_scope('model', reuse=True, initializer=initializer):
            valid_model = InteractiveSeq2SeqModel(train_config, mode='EVAL')

        tf.initialize_all_variables().run()
        start_decay_it = train_config.max_epoch_iter
        prev_valid_loss = np.finfo(np.float32).max

        for it in xrange(train_config.max_epoch_iter):

            print('Training Iter {0}:'.format(it))
            lr_decay = train_config.lr_decay ** max(it - start_decay_it, 0.0)
            cur_lr = train_config.learning_rate * lr_decay
            print('current learning rate: {:.5f}'.format(cur_lr))

            # Resets the learning rate.
            model.assign_lr(session, cur_lr)

            # Runs the training for one epoch.
            train_loss = run_epoch(model, session, train_data, model.train_op,
                                   PAD_ID=train_config.PAD_ID, verbose=True)
            print('Training loss: {:.3f}'.format(train_loss))

            if valid_data is None or it < train_config.skip_valid_iter:
                continue

            # Evaluates the current model.
            cur_valid_loss = run_epoch(valid_model, session, valid_data, tf.no_op(),
                                       PAD_ID=train_config.PAD_ID, verbose=False)
            print('Validation loss: {:.3f}'.format(cur_valid_loss))
            if prev_valid_loss < cur_valid_loss:
                # Second time encounter degenrate performance.
                if start_decay_it < train_config.max_epoch_iter:
                    break

                print('Restoring model')
                ckpt = tf.train.get_checkpoint_state(train_config.model_dir)
                model.saver.restore(session, ckpt.model_checkpoint_path)
                start_decay_it = it
            else:
                print('Saving model')
                model.saver.save(
                    session, os.path.join(train_config.model_dir, 'model.ckpt'),
                    global_step=it)
            prev_valid_loss = cur_valid_loss


def main():
    # Configs training parameters.
    # All these variables can be configurable.
    word_embed = None
    processed_word_embed_fn = None
    if processed_word_embed_fn is not None:
        print('Loading pretrain word embeddings!')
        with open(processed_word_embed_fn) as fin:
            word_embed = pickle.load(fin).astype(np.float32)

    data_dir = 'data'
    train_data_basefn = 'sample_data_pair.txt'
    validate_data_basefn = 'sample_data_pair.txt'

    train_data_fn = os.path.join(data_dir, train_data_basefn)
    validate_data_fn = os.path.join(data_dir, validate_data_basefn)

    out_basedir = 'vars'
    if not os.path.exists(out_basedir):
        os.mkdir(out_basedir)
    model_dir = os.path.join(out_basedir, 'model_dir')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    train_config = config_param()
    train_config.model_dir = model_dir

    train_data = helper.load_pair_idx_data(
        train_data_fn, GO_ID=train_config.GO_ID)
    validate_data = helper.load_pair_idx_data(
        validate_data_fn, GO_ID=train_config.GO_ID)

    # Prints out the configuration.
    print(train_config)
    train_model(train_data, validate_data, train_config, word_embed)


if __name__ == '__main__':
    main()
