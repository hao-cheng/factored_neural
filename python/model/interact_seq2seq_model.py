#!/usr/bin/env python
"""This implementes the factored neural network model.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow.python.ops.seq2seq import attention_decoder
from tensorflow.python.ops.seq2seq import sequence_loss_by_example


linear = tf.nn.rnn_cell._linear


class InteractiveSeq2SeqModel(object):

    def __init__(self, config, mode='TRAIN', loaded_word_embed=None):
        """Builds the computing graph and initializes all variabels.

        Args:
            config: Configuration object contains all model configuration.
            mode: String from {'TRAIN', 'EVAL', 'INFER'}.
            loaded_word_embed: A numpy array of pretrained word embedding.
        """
        # Initilizes model parameters.
        self.batch_size = batch_size = config.batch_size
        self.vocab_size = vocab_size = config.vocab_size
        self.embed_dim = embed_dim = config.embed_dim
        self.hidden_dim = hidden_dim = config.hidden_dim
        self.num_hiddens = num_hiddens = config.num_hiddens
        self.num_modes = num_modes = config.num_modes
        self.mode_dim = mode_dim = config.mode_dim
        self.cmt_seq_len = cmt_seq_len = config.cmt_seq_len
        self.reply_seq_len = reply_seq_len = config.reply_seq_len
        # Objective weight for reply language modeling.
        self.alpha = alpha = config.alpha

        # Initializes placeholders for inputs.
        self.comment_inputs = []
        self.comment_weights = []
        self.reply_inputs = []
        self.reply_weights = []

        self._lr = tf.Variable(0.0, trainable=False)

        for i in xrange(cmt_seq_len):
            self.comment_inputs.append(
                tf.placeholder(tf.int32,
                               name='comment_input_{0}'.format(i),
                               shape=[batch_size]))
            self.comment_weights.append(
                tf.placeholder(tf.float32,
                               name='comment_weight_{0}'.format(i),
                               shape=[batch_size]))
        for i in xrange(reply_seq_len):
            self.reply_inputs.append(
                tf.placeholder(tf.int32,
                               name='reply_input_{0}'.format(i),
                               shape=[batch_size]))
            self.reply_weights.append(
                tf.placeholder(tf.float32,
                               name='reply_weight_{0}'.format(i),
                               shape=[batch_size]))

        self.comment_embeds = []
        self.mix_mode_embeds = []
        self.mode_probs = []
        self.init_reply_embed = []

        # Initlize mode_rnn.
        if mode == 'TRAIN' and config.keep_prob < 1.0:
            mode_rnn = tf.nn.rnn_cell.MultiRNNCell([
                tf.nn.rnn_cell.DropoutWrapper(
                    tf.nn.rnn_cell.BasicLSTMCell(
                        hidden_dim, forget_bias=config.forget_bias,
                        state_is_tuple=True),
                    output_keep_prob=config.keep_prob)
                for _ in xrange(num_hiddens)], state_is_tuple=True)
        else:
            mode_rnn = tf.nn.rnn_cell.MultiRNNCell([
                tf.nn.rnn_cell.BasicLSTMCell(
                    hidden_dim, forget_bias=config.forget_bias,
                    state_is_tuple=True)
                for _ in xrange(num_hiddens)], state_is_tuple=True)

        # Defines the modes.
        batch_mode_inds = tf.constant([range(num_modes)
                                       for _ in range(batch_size)])

        # Defines the embeddings on CPU.
        with tf.device('/cpu:0'):
            mode_embedding = tf.get_variable(
                'mode_embedding',
                [num_modes, mode_dim], dtype=tf.float32)
            att_mode_vecs = tf.nn.embedding_lookup(
                mode_embedding, batch_mode_inds)
            att_states = tf.reshape(
                att_mode_vecs, [-1, num_modes, 1, mode_dim])

        att_mode_weight = tf.get_variable('att_mode_weight',
                                          [1, 1, mode_dim, hidden_dim])

        mode_feat = tf.nn.conv2d(
            att_states, att_mode_weight,
            [1, 1, 1, 1], 'SAME')
        att_v = tf.get_variable('att_v', [hidden_dim])

        def single_attention(query):
            with tf.variable_scope('attention_mlp'):
                y = linear(query, hidden_dim, True)
                y = tf.reshape(y, [-1, 1, 1, hidden_dim])
                s = tf.reduce_sum(att_v * tf.tanh(mode_feat + y), [2, 3])
                a_score = tf.nn.softmax(s)
                weighted_sum = tf.reduce_sum(
                    tf.reshape(a_score, [-1, num_modes, 1, 1]) * att_states,
                    [1, 2])
                a_score = tf.reshape(a_score, [-1, num_modes])
                weighted_sum = tf.reshape(weighted_sum, [-1, mode_dim])
            return a_score, weighted_sum

        with tf.device('/cpu:0'):
            if loaded_word_embed is None:
                embed_weight = tf.get_variable('word_embedding',
                                               [vocab_size, embed_dim])
            else:
                pretrain_word_embed = tf.constant(loaded_word_embed)
                embed_weight = tf.get_variable('word_embedding',
                                               initializer=pretrain_word_embed)

        cmt_state = mode_rnn.zero_state(batch_size, tf.float32)
        c_prev, cell_output = cmt_state[0]

        # Computes the residual value of content and global modes.
        att_proj_weight = tf.get_variable('att_proj_weight',
                                          [mode_dim, hidden_dim])
        att_probs, attns = single_attention(cell_output)
        cell_output += tf.matmul(attns, att_proj_weight)
        cmt_state = [tf.nn.rnn_cell.LSTMStateTuple(c_prev, cell_output)]

        mode_rnn_cell_output = []
        mode_probs = []
        lm_logits = []

        with tf.variable_scope('mode_rnn'):
            for i, cmt_in in enumerate(self.comment_inputs):
                if i > 0: tf.get_variable_scope().reuse_variables()
                cmt_embeds = tf.reshape(
                    tf.nn.embedding_lookup(embed_weight, cmt_in),
                    [batch_size, embed_dim])

                cell_output, cmt_state = mode_rnn(cmt_embeds, cmt_state)
                mode_rnn_cell_output.append(cell_output)
                att_probs, attns = single_attention(cell_output)

                c_prev, _ = cmt_state[0]
                cell_output += tf.matmul(attns, att_proj_weight)

                cmt_state = [tf.nn.rnn_cell.LSTMStateTuple(c_prev, cell_output)]

                with tf.variable_scope('attention_projection'):
                    attention_proj = linear(cell_output, vocab_size, True)

                lm_logits.append(attention_proj)
                mode_probs.append(att_probs)
                if mode == 'INFER':
                    self.mix_mode_embeds.append(attns)

        if mode == 'INFER':
            self.comment_embeds = mode_rnn_cell_output
            self.mode_probs = mode_probs

        top_states = [tf.reshape(e, [-1, 1, mode_rnn.output_size])
                      for e in mode_rnn_cell_output]
        states_for_reply_rnn = tf.concat(1, top_states)

        reply_embeds = [
            tf.reshape(tf.nn.embedding_lookup(embed_weight, reply_i),
                       [batch_size, embed_dim]) for reply_i in self.reply_inputs[:-1]]

        # Initlize reply_rnn.
        if mode == 'TRAIN' and config.keep_prob < 1.0:
            reply_rnn = tf.nn.rnn_cell.MultiRNNCell([
                tf.nn.rnn_cell.DropoutWrapper(
                    tf.nn.rnn_cell.BasicLSTMCell(
                        hidden_dim, forget_bias=config.forget_bias,
                        state_is_tuple=True),
                    output_keep_prob=config.keep_prob)
                for _ in xrange(num_hiddens)], state_is_tuple=True)
        else:
            reply_rnn = tf.nn.rnn_cell.MultiRNNCell([
                tf.nn.rnn_cell.BasicLSTMCell(
                    hidden_dim, forget_bias=config.forget_bias,
                    state_is_tuple=True)
                for _ in xrange(num_hiddens)], state_is_tuple=True)

        reply_rnn_output, reply_rnn_final_state = attention_decoder(
            reply_embeds, cmt_state, states_for_reply_rnn, reply_rnn)

        if mode == 'INFER':
            self.init_reply_embed = reply_rnn_output[0]

        # Computes the language model loss for the comment.
        comment_targets = [cc for cc in self.comment_inputs[1:]]
        lm_loss = tf.reduce_sum(sequence_loss_by_example(
            lm_logits[:-1], comment_targets, self.comment_weights[1:]))

        gen_logits = []
        with tf.variable_scope('gen_logit_projection'):
            for i, rnn_out in enumerate(reply_rnn_output):
                if i > 0: tf.get_variable_scope().reuse_variables()
                logits = linear(rnn_out, vocab_size, True)
                gen_logits.append(logits)

        # Computes the lanuage model loss for the reply.
        reply_targets = [tt for tt in self.reply_inputs[1:]]
        gen_loss = tf.reduce_sum(sequence_loss_by_example(
            gen_logits, reply_targets, self.reply_weights[1:]))

        loss = lm_loss + alpha * gen_loss
        self.total_loss = loss

        self.saver = tf.train.Saver(tf.all_variables())

        if mode != 'TRAIN':
            return

        tvars = tf.trainable_variables()
        grads = tf.gradients(loss, tvars)

        if config.opt_method == 'SGD':
            optimizer = tf.train.GradientDescentOptimizer(self._lr)
        elif config.opt_method == 'AdaDelta':
            optimizer = tf.train.AdadeltaOptimizer(self._lr)
        elif config.opt_method == 'Adam':
            optimizer = tf.train.AdamOptimizer(self._lr)
        else:
            ValueError('Unknown optimizer {}'.format(config.opt_method))
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def assign_lr(self, session, lr_val):
        session.run(tf.assign(self._lr, lr_val))
