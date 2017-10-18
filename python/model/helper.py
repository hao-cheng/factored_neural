#!/usr/bin/env python
"""This defines data processing and configuration helper functions.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np


# Defins constant variables
UNK_TOKEN = '<UNK>'
EOS_TOKEN = '<EOS>'
PAD_TOKEN = '<PAD>'
GO_TOKEN = '<GO>'


class TrainConfig(object):
    def __init__(self):
        self.batch_size = 1
        self.vocab_size = 1
        self.embed_dim = 1
        self.hidden_dim = 1
        self.num_hiddens = 0
        self.num_modes = 0
        self.mode_dim = 0
        self.cmt_seq_len = 0
        self.reply_seq_len = 0
        self.alpha = 0.1
        self.initializer = 'Gaussian'
        self.stddev = 0.1
        self.scale = 0.1
        self.opt_method = 'Adam'
        self.learning_rate = 5e-3
        self.keep_prob = 1.0
        self.forget_bias = 0.1
        self.model_dir = 'SRS-GO/data/model'
        self.rand_seed = 0
        self.skip_valid_iter = 0

    def __str__(self):
        to_print = [
            16*'=',
            'Current config:',
            'batch_size: {0}'.format(self.batch_size),
            'vocab_size: {0}'.format(self.vocab_size),
            'embed_dim: {0}'.format(self.embed_dim),
            'hidden_dim: {0}'.format(self.hidden_dim),
            'num_hiddens: {0}'.format(self.num_hiddens),
            'num_modes: {0}'.format(self.num_modes),
            'mode_dim: {0}'.format(self.mode_dim),
            'cmt_seq_len: {0}'.format(self.cmt_seq_len),
            'reply_seq_len: {0}'.format(self.reply_seq_len),
            'alpha: {0}'.format(self.alpha),
            'initializer: {0}'.format(self.initializer),
            'opt_method: {0}'.format(self.opt_method),
            'learning_rate: {0}'.format(self.learning_rate),
            'keep_prob: {0}'.format(self.keep_prob),
            'model_dir: {0}'.format(self.model_dir),
            'rand_seed: {0}'.format(self.rand_seed),
            'skip_valid_iter {0}'.format(self.skip_valid_iter)
        ]
        if self.initializer == 'Uniform':
            to_print.append('Uniform scale: {0}'.format(self.scale))
        else:
            to_print.append('Gaussian stddev: {0}'.format(self.stddev))
        to_print.append(16*'=')
        return '\n'.join(to_print)


def load_pair_idx_data(data_filename, GO_ID=3):
    """Loads indexed data.

    Args:
        data_filename: String for filename with comment and reply pair.
        GO_ID: Integer for the <GO> symbol index.

    Returns:
        tuple_data: List of (comment, reply, feature, label) tuple.
    """
    tuple_data = []
    with open(data_filename) as fin:
        counter = 0
        for line in fin:
            cmt, reply = line.strip().split('\t')
            counter += 1
            if counter % 1000 == 0:
                print('reading data lien {0}'.format(counter))
            cmt_ids = [int(i) for i in cmt.strip().split()]
            reply_ids = [GO_ID] + [int(i) for i in reply.strip().split()]
            tuple_data.append((cmt_ids, reply_ids))
    return tuple_data


def get_batch_data(tuple_data, cmt_seq_len, reply_seq_len, batch_size, PAD_ID=0):
    """Gets a batch of data.

    Args:
        tuple_data: List of tuples (comment, reply).
        cmt_seq_len: Integer for the max len of comment.
        reply_seq_len: Integer for the max len of reply.
        batch_size: Integer for the batch size.

    Returns:
        batch_cmt_inputs: 2D list of comment id-tokens.
        batch_reply_inputs: 2D list of reply id-tokens.
        batch_cmt_weights: 2D list of comment weights,
                0 if token == PAD, else 1.
        batch_reply_weights: 2D list of reply weights,
                0 if token == PAD, else 1.
        cmt_lens: List of each comment length.
        reply_lens: List of each reply length.
    """

    cmt_inputs, cmt_lens, reply_inputs, reply_lens = [], [], [], []
    chosen_indices = np.random.choice(len(tuple_data), batch_size)
    for chosen_idx in chosen_indices:
        cinputs, rinputs = tuple_data[chosen_idx]

        # Gets the length info.
        cmt_lens.append(len(cinputs))
        reply_lens.append(len(rinputs))

        # Pads the comment if needed.
        cmt_pad = [PAD_ID] * (cmt_seq_len - len(cinputs))
        cmt_inputs.append(list(cinputs + cmt_pad))

        # Pads the reply if needed.
        reply_pad = [PAD_ID] * (reply_seq_len - len(rinputs))
        reply_inputs.append(list(rinputs + reply_pad))

    (batch_cmt_inputs, batch_reply_inputs,
     batch_cmt_weights, batch_reply_weights) = [], [], [], []

    for idx in xrange(cmt_seq_len):
        batch_cmt_inputs.append(
            np.array([cmt_inputs[batch_idx][idx]
                      for batch_idx in xrange(batch_size)], dtype=np.int32))
        batch_weight = np.ones(batch_size, dtype=np.float32)

        for batch_idx in xrange(batch_size):
            if idx < cmt_seq_len - 1:
                target = cmt_inputs[batch_idx][idx]
            if idx == cmt_seq_len - 1 or target == PAD_ID:
                batch_weight[batch_idx] = 0.0
        batch_cmt_weights.append(batch_weight)

    for idx in xrange(reply_seq_len):
        batch_reply_inputs.append(
            np.array([reply_inputs[batch_idx][idx]
                      for batch_idx in xrange(batch_size)], dtype=np.int32))

        batch_weight = np.ones(batch_size, dtype=np.float32)

        for batch_idx in xrange(batch_size):
            if idx < reply_seq_len - 1:
                target = reply_inputs[batch_idx][idx]
            if idx == reply_seq_len - 1 or target == PAD_ID:
                batch_weight[batch_idx] = 0.0
        batch_reply_weights.append(batch_weight)

    return (batch_cmt_inputs, batch_reply_inputs,
            batch_cmt_weights, batch_reply_weights,
            cmt_lens, reply_lens)
