
import theano.tensor as T
from layers import *

def network(img_size=84, n_actions=7, n_channels=1):
    l0 = DataLayer()
    l0 = Layer(l0, fun=lambda x: 1.0 / (1.001 - x))

    l1 = ConvLayer(input_layer=l0,
                   filter_shape=(n_channels, 8, 8, 32),
                   stride=4, pad=0, std=0.01, bias=0.1)
    l1 = PReLULayer(l1)

    l2 = ConvLayer(input_layer=l1,
                   filter_shape=(32, 4, 4, 64),
                   stride=2, pad=0, std=0.01, bias=0.1)
    l2 = PReLULayer(l2)

    l3 = ConvLayer(input_layer=l2,
                   filter_shape=(64, 3, 3, 64),
                   stride=1, pad=0, std=0.01, bias=0.1)
    l3 = PReLULayer(l3)

    l4 = Layer(l3, fun=lambda x: T.flatten(x, 2))
    l4 = DenseLayer(input_layer=l4,
                    n_in=64 * 7 * 7,
                    n_out=512, std=.005, bias=0.1)
    l4 = PReLULayer(l4)
    l4 = DropoutLayer(input_layer=l4, dropout=0.5)

    l5 = DenseLayer(input_layer=l4,
                    n_in=512,
                    n_out=n_actions, std=.005, bias=0.1)

    return l5, [l4, l3, l2, l1, l0]
