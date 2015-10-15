"""
Code for deep Q-learning as described in:

Playing Atari with Deep Reinforcement Learning
NIPS Deep Learning Workshop 2013

and

Human-level control through deep reinforcement learning.
Nature, 518(7540):529-533, February 2015


Author of Lasagne port: Nissan Pow
Modifications: Nathan Sprague
Modifications: Liu Bingyuan
"""
import lasagne
import numpy as np
import theano
import theano.tensor as T


class DeepQLearner:
    """
    Deep Q-learning network using Lasagne.
    """

    def __init__(self, input_width, input_height, num_actions,
                 num_frames, discount, learning_rate,momentum,
                 batch_size, ):

        self.input_width = input_width
        self.input_height = input_height
        self.num_actions = num_actions
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.discount = discount
        self.lr = learning_rate
        self.momentum = momentum

        self.update_counter = 0

        states = T.tensor4('states')
        next_states = T.tensor4('next_states')
        rewards = T.col('rewards')
        actions = T.icol('actions')
        terminals = T.icol('terminals')

        self.states_shared = theano.shared(
            np.zeros((batch_size, num_frames, input_height, input_width),
                     dtype=theano.config.floatX))

        self.next_states_shared = theano.shared(
            np.zeros((batch_size, num_frames, input_height, input_width),
                     dtype=theano.config.floatX))

        self.rewards_shared = theano.shared(
            np.zeros((batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))

        self.actions_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))

        self.terminals_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))

        from nn import network
        n, layers = network(n_channels=num_frames,
                            img_size=input_width, n_actions=num_actions)
        self.n = n
        q_vals = n.output(data_layer=states)
        next_q_vals = n.output(data_layer=next_states)
        next_q_vals = theano.gradient.disconnected_grad(next_q_vals)
        next_q_vals = T.minimum(0, next_q_vals)

        layers_samples = [l.output(data_layer=states) for l in layers]
        layers_batchstd = [T.mean(T.std(s, axis=0)) for s in layers_samples]
        w, b = n.weight(), n.bias()
        params = w + b

        target = (rewards +
                  (T.ones_like(terminals) - terminals) *
                  self.discount * T.max(next_q_vals, axis=1, keepdims=True))
        diff = target - q_vals[T.arange(batch_size),
                               actions.reshape((-1,))].reshape((-1, 1))

        loss = T.mean(diff ** 2)

        givens = {
            states: self.states_shared,
            next_states: self.next_states_shared,
            rewards: self.rewards_shared,
            actions: self.actions_shared,
            terminals: self.terminals_shared
        }
        updates = lasagne.updates.rmsprop(loss, params, self.lr)

        if self.momentum > 0:
            updates = lasagne.updates.apply_momentum(updates, None,
                                                     self.momentum)

        self._train = theano.function([], [loss, q_vals], updates=updates,
                                      givens=givens)
        self._batchstd = theano.function([], layers_batchstd,
                                         givens={states: self.states_shared})
        self._sample = theano.function([], layers_samples,
                                       givens={states: self.states_shared})
        self._q_vals = theano.function([states], q_vals,)

    def save(self, p):
        self.n.saveParams(p)

    def load(self, p):
        self.n.loadParams(p)

    def train(self, states, actions, rewards, next_states, terminals):
        """
        Train one batch.

        Arguments:

        states - b x f x h x w numpy array, where b is batch size,
                 f is num frames, h is height and w is width.
        actions - b x 1 numpy array of integers
        rewards - b x 1 numpy array
        next_states - b x f x h x w numpy array
        terminals - b x 1 numpy boolean array (currently ignored)

        Returns: average loss
        """
        self.states_shared.set_value(states)
        self.next_states_shared.set_value(next_states)
        self.actions_shared.set_value(actions)
        self.rewards_shared.set_value(rewards)
        self.terminals_shared.set_value(terminals)
        loss, q_vals = self._train()
        self.update_counter += 1
        return np.sqrt(loss), q_vals

    def q_vals(self, states):
        return self._q_vals(states)

