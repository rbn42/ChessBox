import os.path
import sys
import logging
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.cuda import dnn

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import warnings
warnings.filterwarnings("ignore")

rng = np.random.RandomState()
srng = RandomStreams()


class Layer:

    def __init__(self, input_layer, fun=None):
        self.input_layer = input_layer
        self.n_channel = self.input_layer.n_channel
        self.fun = fun

    def output(self,   *args, **kwargs):
        assert not None == self._output
        assert not self == self.input_layer
        input = self.input_layer.output(*args, **kwargs)
        return self._output(input=input,  *args, **kwargs)

    def _output(self, input,   *args, **kwargs):
        return self.fun(input)

    def weight(self):
        return self.input_layer.weight()

    def bias(self):
        return self.input_layer.bias()

    def getParams4SL(self, count):
        return [], count

    def saveParams(self, path, count=0):
        if not os.path.exists(path):
            os.makedirs(path)
        params, count = self.getParams4SL(count)
        self.input_layer.saveParams(path, count)
        for n, p in params:
            n = os.path.join(path, n) + '.npy'
            np.save(n, p.get_value())

    def loadParams(self, path, count=0):
        params, count = self.getParams4SL(count)
        self.input_layer.loadParams(path, count)
        for n, p in params:
            n = os.path.join(path, n) + '.npy'
            if os.path.exists(n):
                p.set_value(np.load(n))
            else:
                print('parameters file %s not found' % n)


class DataLayer:

    def __init__(self,  n_channel=3):
        self.n_channel = n_channel

    def output(self, data_layer,  *args, **kwargs):
        return data_layer

    def weight(self):
        return []

    def bias(self):
        return []

    def saveParams(self, path, count=0):
        return

    def loadParams(self, path, count=0):
        return


class ConvLayer(Layer):

    def __init__(self, input_layer,  filter_shape, stride, pad, bias, std=0.01, fixed=False, name=None):
        self.n_in = filter_shape[0]
        self.n_out = filter_shape[3]
        self.n_channel = self.n_out
        self.input_layer = input_layer
        self.subsample = (stride, stride)
        self.pad = pad
        self.filter_shape = np.asarray(filter_shape)
        self.W = generateWeight(self.filter_shape, std=std)
        self.b = generateWeight(self.filter_shape[3], init=bias, std=0)
        self.fixed = fixed
        self.name = name

    def _output(self, input,  *args, **kwargs):
        input_shuffled = input
        W_shuffled = self.W.dimshuffle(3, 0, 1, 2)
        conv_out = dnn.dnn_conv(img=input_shuffled,
                                kerns=W_shuffled,
                                subsample=self.subsample,
                                border_mode=self.pad)
        conv_out = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        out = conv_out
        return out

    def getParams4SL(self, count):

        if None == self.name:
            w = 'conv%.3d_w' % count
            b = 'conv%.3d_b' % count
            count += 1
        else:
            w = self.name + '_w'
            b = self.name + '_b'
        w += '_%dx%d' % (self.n_in, self.n_out)
        b += '_%dx%d' % (self.n_in, self.n_out)
        return [(w, self.W), (b, self.b)], count

    def weight(self):
        if self.fixed:
            return [] + self.input_layer.weight()
        return [self.W] + self.input_layer.weight()

    def bias(self):
        if self.fixed:
            return [] + self.input_layer.bias()
        return [self.b] + self.input_layer.bias()


class PReLULayer(Layer):

    def __init__(self, input_layer,  alpha=.25, name=None, fixed=False ):
        self.input_layer = input_layer
        self.n_channel = self.input_layer.n_channel
        affected_channels = getattr(input_layer, 'affected_channels', None)
        self.affected_channels = self.n_channel if None == affected_channels else affected_channels
        assert self.n_channel >= self.affected_channels
        self.alpha = generateWeight(self.affected_channels, init=alpha, std=0)
        self.name = name
        self.fixed = fixed

    def _output(self, input,  *args, **kwargs):
        ndim = input.ndim
        if 4 == ndim:
            filter_shape = 1, self.affected_channels, 1, 1
        elif 2 == ndim:
            filter_shape = 1, self.affected_channels
        k = (self.alpha - 1).reshape(filter_shape)
        if self.affected_channels == self.n_channel:
            return input + T.minimum(0, input) * k
        else:
            affected = input[:, :self.affected_channels]
            unaffected = input[:, self.affected_channels:]
            affected = affected + T.minimum(0, affected) * k
            return T.concatenate([affected, unaffected], axis=1)

    def getParams4SL(self, count):
        if None == self.name:
            a = 'prelu%.3d_a' % count
            count += 1
        else:
            a = self.name + '_a'
        a += '_' + str(self.affected_channels)
        return [(a, self.alpha)], count

    def weight(self):
        if self.fixed:
            return [] + self.input_layer.weight()
        return [self.alpha] + self.input_layer.weight()


class DropoutLayer(Layer):

    def __init__(self, input_layer, dropout):
        self.input_layer = input_layer
        self.dropout = dropout
        self.n_channel = self.input_layer.n_channel

    def _output(self, input, dropout_active=True,  *args, **kwargs):
        if not dropout_active:
            return input
        if self.dropout < 0:
            return input
        retain_prob = 1 - self.dropout
        out = input / retain_prob * srng.binomial(size=input.shape,
                                                  p=retain_prob,
                                                  dtype='int32').astype('float32')
        return out


class DenseLayer(Layer):

    def __init__(self, input_layer, n_in, n_out,
                 weight_np=None, bias_np=None,
                 std=.005, bias=1, fixed=False, name=None):
        self.input_layer = input_layer
        self.n_out = n_out
        self.n_channel = n_out
        self.n_in = n_in

        if None == weight_np:
            self.W = generateWeight((n_in, n_out), std=std)
        else:
            self.W = theano.shared(weight_np)
        if None == bias_np:
            self.b = generateWeight(n_out, init=bias, std=0)
        else:
            self.b = theano.shared(bias_np)
        self.name = name
        self.fixed = fixed

    def _output(self, input,  *args, **kwargs):
        out = T.dot(input, self.W) + self.b
        return out

    def getParams4SL(self, count):
        if None == self.name:
            w = 'dense%.3d_w' % count
            b = 'dense%.3d_b' % count
            count += 1
        else:
            w = self.name + '_w'
            b = self.name + '_b'
        w += '_%dx%d' % (self.n_in, self.n_out)
        b += '_%dx%d' % (self.n_in, self.n_out)
        return [(w, self.W), (b, self.b)], count

    def weight(self):
        if self.fixed:
            return [] + self.input_layer.weight()
        return [self.W] + self.input_layer.weight()

    def bias(self):
        if self.fixed:
            return [] + self.input_layer.bias()
        return [self.b] + self.input_layer.bias()


def generateWeight(w_shape,  std, init=0):
    if std != 0:
        np_values = np.asarray(
            rng.normal(init, std, w_shape), dtype=theano.config.floatX)
    else:
        np_values = np.cast[theano.config.floatX](
            init * np.ones(w_shape, dtype=theano.config.floatX))
    return theano.shared(np_values)
