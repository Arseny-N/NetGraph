
from pathlib import Path

class FileGraphBase:
    def __init__(self):
        self.stats = {
            'nodes' : 0, 
            'edges' : 0
        }


class GMLGraph(FileGraphBase):

    def __init__(self, file):

        super().__init__()

        self.file = open(Path(file).expanduser(), 'w')
        self.write( "graph [\n\tdirected 1\n")
            
    def write(self, str):
        self.file.write(str)

    def close(self):
        self.write("]\n")
        self.file.close()

    def add_nodes_from(self, nodes, **props):

        props_str = '\n'.join( "\t\t" + key + " " + str(val) for key, val in props.items())

        for node_id, attrs in nodes:

            attrs_str = [] 
            for key, val in attrs.items():
                if key == 'pos':
                    attrs_str.append("\t\tgraphics [\n" + \
                        f"\t\t\tx {val[0]}\n" + \
                        f"\t\t\ty {val[1]}\n" + \
                        f"\t\t\tz {val[2]}\n" + \
                        "\t\t]"
                        )
                else:
                    attrs_str.append("\t\t" + key + " " + str(val))
            attrs_str = '\n'.join(attrs_str)

            self.write(f"\tnode [\n\t\tid {node_id}\n{attrs_str}\n{props_str}\n\t]\n")
                
    def add_edges_from(self, edges, **props):

        props_str = '\n'.join( "\t\t" + key + " " + str(val) for key, val in props.items())

        for source, target, attrs in edges:
            attrs_str = '\n'.join( "\t\t" + key + " " + str(val) for key, val in attrs.items() )
            self.write(f"\tedge [\n\t\tsource {source}\n\t\ttarget {target}\n{attrs_str}\n{props_str}\n\t]\n")




class CSVGraph(FileGraphBase):
    
    def __init__(self, dir, prefix,  node_props, edge_props):
        super().__init__()

        assert 'node_id' == node_props[0]
        assert 'Source' == edges_props[0] and 'Target' == edges_props[1]
        
        self.nodes_file = open(Path(dir).expanduser() / (prefix + 'nodes.csv'))
        self.nodes_file.write(','.join(map(lambda x: "'" + x + "'", node_props)))

        self.edges_file = open(Path(dir).expanduser() / (prefix + 'edges.csv'))
        self.edges_file.write(','.join(map(lambda x: "'" + x + "'", edge_props)))

    def add_nodes_from(self, nodes, **props):
        raise NotImplementedError()
    def add_edges_from(self, edges, **props):
        raise NotImplementedError()


import h5py

class HDF5Graph(FileGraphBase):
    def __init__(self, file, node_attrs, edge_attrs):

        self.f = h5py.File(Path(file).expanduser(), 'w')

        self.edges_ds = self.f.create_dataset('edges', (10_000, 2), chunks=True, maxshape=(None, 2), dtype="i")
        self.edges_attrs_ds = self.f.create_dataset('edges_attr', (10_000, len(edge_attrs)), chunks=True, maxshape=(None, len(edge_attrs)), dtype="f")
        self.edges_num = 0

        self.nodes_ds = self.f.create_dataset('nodes', (10_000, len(node_attrs)), chunks=True, maxshape=(None, len(node_attrs)), dtype="f")


        self.node_attrs_to_ix = { attr : ix for ix, attr in enumerate(node_attrs) }
        self.edge_attrs_to_ix = { attr : ix for ix, attr in enumerate(edge_attrs) }

        self.tensor_name_to_ix = {}



    def add_nodes_from(self, nodes, **props):

        for node_id, attrs in nodes:

            n, m = self.nodes_ds.shape
            if n <= node_id:
                self.nodes_ds.resize(n + 10_000, axis=0)
                


            for name, attr in attrs.items():
                self.nodes_ds[node_id, self.node_attrs_to_ix[name]] = attr

            for name, attr in props.items():

                if name == 'tensor_name':
                    if name in self.tensor_name_to_ix:
                        attr = self.tensor_name_to_ix[attr]
                    else:
                        attr = self.tensor_name_to_ix[attr] = len(self.tensor_name_to_ix)

                self.nodes_ds[node_id, self.node_attrs_to_ix[name]] = attr
            
                
    def add_edges_from(self, edges, **props):

        for source, target, attrs in edges:

            self.edges_num += 1
            edge_id = self.edges_num

            n, m = self.edges_ds.shape
            if n <= edge_id:
                self.edges_ds.resize(n + 10_000, axis=0)
                self.edges_attrs_ds.resize(n + 10_000, axis=0)

            self.edges_ds[edge_id, 0] = source
            self.edges_ds[edge_id, 1] = target


            for name, attr in attrs.items():
                self.edges_attrs_ds[edge_id, self.edge_attrs_to_ix[name]] = attr

            for name, attr in props.items():
                self.edges_attrs_ds[edge_id, self.edge_attrs_to_ix[name]] = attr