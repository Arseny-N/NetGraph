
from ...keras_frontend import KerasFrontend

def lenet(ng):

    
    layers = KerasFrontend(ng)

    x = layers.Input(name='conv2d_1_input')

    x = layers.Conv2D(6, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = layers.Conv2D(16, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(120, activation = 'relu', kernel_initializer='he_normal' )(x)
    x = layers.Dense(84, activation = 'relu', kernel_initializer='he_normal' )(x)
    x = layers.Dense(10, activation = 'softmax', kernel_initializer='he_normal' )(x)        

    return ng



from netgraph.graph import HDF5Graph, GMLGraph

# tulip python console seems to be broken in ubuntu :(
class GMLGraph1(GMLGraph):

    def __init__(self, file):

        super().__init__(file)

        self.nodes_acts = {}
            

    def add_nodes_from(self, nodes, **props):

        def _nodes():
            for node_id, attrs in nodes:
                self.nodes_acts[node_id] = attrs['activation']
                yield (node_id, attrs)

        super().add_nodes_from(_nodes(), **props)
        
                
    def add_edges_from(self, edges, **props):

        def _edges():
            for source, target, attrs in edges:

                attrs['weight_src'] = self.nodes_acts[source]
                attrs['weight_tgt'] = self.nodes_acts[target]
                attrs['weight_mean'] = (self.nodes_acts[target] + self.nodes_acts[source])/2
                
                yield (source, target, attrs) 

        super().add_edges_from(_edges(), **props)

def main():

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

    from netgraph.netgraph import NetGraph
    from netgraph.graph import HDF5Graph, GMLGraph
    from netgraph.nets.data import data

    import numpy as np
    import pandas as pd

    from keras import backend as K
       
   
    from keras.models import load_model

    model = load_model('/nfs/home/anerinovsky/ng/weights/lenet.h5')
    (x_train, x_train_raw, y_train, y_train_raw), (x_test, x_test_raw, y_test, y_test_raw) = data('cifar10')

    x, x_raw, y = x_test[:1], x_test_raw[:1], y_test_raw[0]

    sess = K.get_session()
    input_name = 'conv2d_1_input'

    # graph = HDF5Graph(
    #     file='~/data/ng/lenet-mnist.hdf5',
    #     node_attrs=['activation', 'tensor_name'],
    #     edge_attrs=['comp_weight', 'mult_weight']
    # )

    graph = GMLGraph1(file='~/data/ng/lenet-mnist.gml')

    ng = NetGraph(sess=sess, 
        feed_dict={ 
            sess.graph.get_tensor_by_name(input_name + ':0') : x 
        }, 
        graph=graph
    )

    lenet(ng)

    # ng.graph.



if __name__ == '__main__':
    main()