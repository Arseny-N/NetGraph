class Tensor:
    def __init__(self, data, ng, name, nodes):
        self.data = data
        self.ng = ng
        self.name = name
        self.nodes = nodes.reshape(data.shape)

        
    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    @property
    def ndim(self):
        return self.data.ndim

    def node_list(self):
        return self.nodes.ravel().tolist()