from torch_geometric.data import Data


class CochainData(Data):
    def __init__(self, lower_index = None, upper_index = None, lower_values = None, upper_values = None,
                 x_edge = None, label = None):
        super().__init__(y = label)
        self.lower_index = lower_index
        self.upper_index = upper_index
        self.lower_values = lower_values
        self.upper_values = upper_values
        self.x_edge = x_edge
        self.num_triangles = None

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return self.num_nodes
        elif 'index' in key:
            return self.x_edge.shape[0]
        else:
            return 0

    def unpack_cochain(self):
        return self.x_edge, self.lower_index, self.upper_index, self.lower_values, self.upper_values



