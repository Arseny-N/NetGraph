

# NetGraph: A framework for graph based neural network exploration 

This package allows to create a graph from a neural network.

# Example: Building LeNet


```python

from netgraph.keras_frontend import  KerasFrontend
from netgraph.netgraph import NetGraph
from netgraph.graph import GMLGraph
from netgraph.data_extractor import TFDataExtractor

from keras.models import load_model
from netgraph.nx.nets.lenet import data


model = load_model('weights/lenet-mnist.hdf5')

(x_train, x_train_raw, y_train), (x_test, x_test_raw, y_test) = data('mnist')

# Output the graph to a file wihout storing it
graph = GMLGraph(file='lenet-mnist.gml')

# Extract the tensors and weights from a tensorflow graph 
data_extractor = TFDataExtractor(
        sess = K.get_session(), 
        feed_dict = { 
            K.get_session().graph.get_tensor_by_name('conv2d_1_input:0') : x_test[:1]
        }
    )

# Build the NetGraph
ng = NetGraph(graph=graph, data_extractor=data_extractor)

# Create the graph
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


```

Visualization of the resulting network in tulpan.

![image](images/net.jpg)

