from scipy.sparse import lil_matrix


from tqdm.autonotebook import tqdm

import numpy as np

from functools import wraps, partial

from itertools import product, repeat, starmap


import networkx as nx

from collections.abc import Iterable

from .data_extractor import TFDataExtractor
from .graph import GMLGraph
from .tensor import Tensor
from .layout import layout_tensor

class NetGraph:
    
    def __init__(self, sess=None, 
        data_extractor=None, 
        feed_dict=None, graph=None, 
        need_layout=True):

        if graph is None:
            graph = nx.DiGraph()

        self.graph = graph
        self.nodes = {}
        self.tensors = {}
        
        self.weights = {}
        
        self._last_node = 0
        self._anon_tensor_count = 0

        self._max_z = 0
        self.need_layout = need_layout

        if data_extractor is None:
            from keras import backend as K

            if sess is None:
                sess = K.get_session()
            assert feed_dict is not None
            self.extractor = TFDataExtractor(sess, feed_dict)
        else:
            self.extractor = data_extractor
    
    def tensor(self, name):
        if name in self.tensors:
            return self.tensors[name]
        else:
            print(f'Extracting tesor {name}')
            data = self.extractor.extract_tensor(name)
            print(f'Registering tensor {name}')
            return self.register_tensor(data, name)


    def weight(self, name):
        if name in self.weights:
            return self.weights[name]
        else:
            data = self.extractor.extract_weight(name)
            return self.register_weight(data, name)

        
    
    def register_tensor(self, data, name=None):

        if name is None:
            name = self._gen_tensor_name()

        assert name not in self.tensors, f"{name} is already in {self.tensors}"
        
        nodes = self._alloc_nodes(data, name)
        tensor = Tensor(data=data, ng=self, name=name, nodes=nodes)
        
        self.tensors[name] = tensor

        return tensor
    
    def register_weight(self, weight, name):
        assert name not in self.weights, f"{name} is already in {self.weights}"
        
        self.weights[name] = weight
        return weight
    
    def _alloc_nodes(self, data, tensor_name):
        
        b = self._last_node 
        self._last_node += data.size
        
        node_range = (b, self._last_node)
        nodes = np.arange(*node_range) 

        

        if self.need_layout:
            layout = layout_tensor(data.shape, z_offset=self._max_z + 4)
            self._max_z = layout[:, 2].max()
            attrs = starmap(lambda x, p: dict(activation=x, pos=p), zip(data.ravel(), layout))
        else:
            attrs = map(lambda x: dict(activation=x), data.ravel())

        _reg_nodes = zip(nodes, attrs)


        self.graph.add_nodes_from(_reg_nodes, tensor_name=tensor_name)

        return nodes
    
    def _normalize_tensors(self, xs):
        result = []
        for x in xs:

            if isinstance(x, str):
                x = self.tensor(x)

            assert isinstance(x, Tensor), f'{x} is expected to be a Tensor'
            result.append(x)
        return result
    
    def _normalize_weights(self, xs):
        result = []
        for x in xs:
            if isinstance(x, str):
                x = self.weight(x)
            result.append(x)
        return result
    
    
    def _gen_tensor_name(self):
        self._anon_tensor_count += 1
        return f"anon<{self._anon_tensor_count}>"
   
    
    ops = { }
    
    
    def __getattr__(self, attr):
        if attr in self.ops:
            return partial(self.ops[attr], self)
        
        return object.__getattribute__(self, attr)

    def add_edges(self, source, dest, **weights):
        if isinstance(source, Iterable) and \
            not isinstance(dest, Iterable):
            dest = repeat(dest)

        elif isinstance(dest, Iterable) and \
            not isinstance(source, Iterable):

            dest = repeat(source)
        elif not isinstance(dest, Iterable) and \
                not isinstance(source, Iterable):
            dest, source = [dest], [source]


        if weights:
            # weights: dict(w_name1=[w11, w12, ...], w_name2=[w21, w22, ...], ...)
            # need: [dict(w_name1=w11, w_name2=w21), dict(w_name1=w12, w_name2=w22), ...]
            w_names = list(weights.keys())
            ws = zip( *(weights[w_name] for w_name in w_names))
            weights = (dict(zip(w_names, w)) for w in ws)

            edges = zip(source, dest, weights)
        else:
            edges = zip(source, dest, repeat({}))

        self.graph.add_edges_from(edges)


    @classmethod
    def keras_to_file(cls, input, file):
        from keras import backed as K
        from .keras_frontend import KerasFrontend

        graph = GMLGraph(file='lenet-mnist.gml')
        
        data_extractor = TFDataExtractor(
            sess = K.get_session(), 
            feed_dict = { 
                K.get_session().graph.get_tensor_by_name('conv2d_1_input:0') : input
            }
        )


        ng = cls(graph=graph, data_extractor=data_extractor)
        return KerasFrontend(ng)







def register_op(op):

    @wraps(op)
    def wrap(self, data, weights=[], **params):
        
        data = self._normalize_tensors(data)
        weights = self._normalize_weights(weights)
        
        op(self, data, weights, **params)
        
        return data[-1]
    
    NetGraph.ops[op.__name__] = wrap
    return wrap


def register_op_dont_normalize(op):

    @wraps(op)
    def wrap(self, data, weights=[], **params):
        
        op(self, data, weights, **params)
        
        return data[-1]
    
    NetGraph.ops[op.__name__] = wrap
    return wrap







                    

@register_op
def activation(ng, data, weights):
    input, output = data
    ng.add_edges(input.nodes.ravel(), output.nodes.ravel())    
            


@register_op
def relu(ng, data, weights):
    input, output = data
    ng.add_edges(input.nodes.ravel(), output.nodes.ravel())



@register_op
def softmax(ng, data, weights):    
    input, output = data
    ng.add_edges(input.nodes.ravel(), output.nodes.ravel())
       

@register_op
def reshape(ng, data, weights):
    input, output = data
    ng.add_edges(input.nodes.ravel(), output.nodes.ravel())



@register_op
def dense(ng, data, weights):
    
    input, output = data
    weight, bias = weights
    
    batch_size, h_dim = input.shape
    batch_size, o_dim = output.shape
    
    assert weight.shape == (h_dim, o_dim)
    
    for b in range(batch_size):
        for i, o_node in enumerate(output.nodes[b, ...]):

            ng.add_edges(input.nodes.ravel(), o_node, 
                comp_weight=weight[:, i]*input.data[b], 
                mult_weight=weight[:, i])
                




def calculate_padding(input_length, filter_size, padding, stride, dilation=1):
    
    padding = padding.lower()

    assert padding in {'same', 'valid', 'full', 'causal'}
    
    if padding == 'same':
        return ((stride-1)*input_length-stride+filter_size)//2
    elif padding == 'valid':
        return 0


def pad_tensor(ng, input, output=None, padding=(1,1)):

    if isinstance(padding, int):
        (ph_t, ph_b), (pw_l, pw_r) = (padding, padding), (padding, padding)
    elif isinstance(padding[0], int):
        (ph_t, ph_b), (pw_l, pw_r) = ((padding[0], padding[0]), (padding[1], padding[1]))
    else:
        (ph_t, ph_b), (pw_l, pw_r) = padding

    padded_input = np.pad(input.data, ((0,0), (ph_t, ph_b), (pw_l, pw_r), (0,0)), mode='constant')

    if output is None:
        output = ng.register_tensor(padded_input)

    ng.add_edges(input.nodes.ravel(), 
        output.nodes[:, ph_t:output.nodes.shape[1]-ph_b, 
                        pw_l:output.nodes.shape[2]-pw_r, :].ravel())

    return output

def pad_tensor_for_conv(ng, input, kernel_shape, padding, strides):
    
    batch_size, ih, iw, ic = input.shape
    kh, kw = kernel_shape
    
    
    ph, pw = [
        calculate_padding(ih, kh, padding=padding, stride=strides[0]),
        calculate_padding(iw, kw, padding=padding, stride=strides[1]) 
    ]
    
    return pad_tensor(ng, input, padding=(ph, pw))


def to_int_x_int(x):
    if isinstance(x, int):
        return (x, x)
    return x


@register_op_dont_normalize
def maxpool2d(ng, data, weights, pool_size,  padding='valid', strides=None):
    
    input, output = data

    input = ng._normalize_tensors([input])[0]

    pool_size = to_int_x_int(pool_size)
    if strides is None:
        strides = pool_size

    strides = to_int_x_int(strides)

    input = pad_tensor_for_conv(ng, input, pool_size, padding, strides)   


    batch_size, ih, iw, ic = input.shape
    kh, kw = pool_size

    output = ng._normalize_tensors([output])[0]
    

    _, oh, ow, oc = output.shape
    
    
    assert ic == oc

    for b in range(batch_size):
        for i in range(ow):
            for j in range(oh):
                # TODO: vectorize the channel loop
                for ch in range(oc):
                    _i, _j = i*strides[0], j*strides[1]

                    _i_nodes = input.nodes[b, _i:_i+kw, _j:_j+kh, ch].ravel()
                    _input = input.data[b, _i:_i+kw, _j:_j+kh, ch].ravel()
                    m_ix = _input.argmax()

                    ng.add_edges(_i_nodes[m_ix], output.nodes[b, i, j, ch])





@register_op_dont_normalize
def conv2d(ng, data, weights, padding='valid', strides=(1,1)):
    '''
        input : [1,H,W,C]
        kernel: [filter_height, filter_width, 
                    in_channels, out_channels]
    '''
    
    input, output = data

    input = ng._normalize_tensors([input])[0]

    kernel, bias = ng._normalize_weights(weights)

    strides = to_int_x_int(strides)
    input = pad_tensor_for_conv(ng, input, kernel.shape[:2], padding, strides)   

    output = ng._normalize_tensors([output])[0]
        
    kh, kw, kic, koc = kernel.shape
    
    batch_size, oh, ow, oc = output.shape
    
    
    for b in range(batch_size):
        for i in tqdm(range(ow)):
            for j in range(oh):
                for ko in range(koc):
                    
                    _i, _j = i*strides[0], j*strides[1]
                    _i_nodes = input.nodes[b, _i:_i+kw, _j:_j+kh, :]
                    
                    assert _i_nodes.size == kernel[..., ko].size, f"{_i_nodes.size} {kernel[..., ko].size}"
                    
                    w = (kernel[..., ko] * input.nodes[b, _i:_i+kw, _j:_j+kh, :]).ravel() 

                    if bias is not None:
                        w += bias[ko]
                        
                    ng.add_edges(_i_nodes.ravel(), output.nodes[b, i, j, ko],
                            comp_weight=w, mult_weight=kernel[..., ko].ravel(), 
                            # add_weight=bias[ko]
                            )


@register_op
def zero_padding_2d(ng, data, weights, padding=(1,1)):
    input, output = data

    pad_tensor(ng, input, output, padding=padding)

@register_op
def add(ng, data, weights):
    *inputs, output = data
    
    assert all( input.shape == output.shape for input in inputs)

    for input in inputs:
        ng.add_edges(input.nodes.ravel(), output.nodes.ravel())


@register_op
def batch_norm_2d(ng, data, weights, eps=1e-5, axis=-1):

    # https://towardsdatascience.com/pitfalls-of-batch-norm-in-tensorflow-and-sanity-checks-for-training-networks-e86c207548c8
    # https://gluon.mxnet.io/chapter04_convolutional-neural-networks/cnn-batch-norm-scratch.html
    gamma, beta, moving_mean, moving_variance = weights 
    input, output = data
    assert input.ndim == 4
    
    X = input.data

    if axis == 1:
        N, C, H, W = X.shape
        X_hat = (X - moving_mean.reshape((1, C, 1, 1))) * 1.0 / np.sqrt(moving_variance.reshape((1, C, 1, 1)) + eps)
        out = gamma.reshape((1, C, 1, 1)) * X_hat + beta.reshape((1, C, 1, 1))
    elif axis == 3 or axis == -1:
        N, H, W, C = X.shape
        X_hat = (X - moving_mean.reshape((1, 1, 1, C))) * 1.0 / np.sqrt(moving_variance.reshape((1, 1, 1, C)) + eps)
        out = gamma.reshape((1, 1, 1, C)) * X_hat + beta.reshape((1, 1, 1, C))
    else:
        raise NotImplementedError('Axis can be only 1, 3, -1')


    ng.add_edges(input.nodes.ravel(), output.nodes.ravel(), 
        comp_weight=out.ravel())


def assert_shape(tensor, shape):
    if not all(ts == s 
        for ts, s in zip(tensor.shape, shape) 
            if s != None):
        raise ValueError(f"Expecting tensor of shape {shape}, but got tensor of shape {tensor.shape}")

@register_op
def global_average_pooling_2d(ng, data, weights):
    input, output = data

    assert_shape(output, (None, 1, 1, None))

    batch_size, oh, ow, oc = output.shape

    for batch in range(batch_size):
        for ch in range(oc):
            _input = input.nodes[batch, :, :, oc]
            _sum = _input.sum()
            ng.add_edges(input.nodes[batch, :, :, oc].ravel(), output.nodes[batch, 0, 0, oc],
                comp_weight=_input.ravel()/_sum,
                mult_weight=[1/_sum]*_input.size
                )