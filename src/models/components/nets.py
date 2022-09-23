from torch_geometric.nn import GCNConv, global_add_pool as gap, global_mean_pool as gmp

from src.models.components.pool_layers import *


class BaseNet(nn.Module):
    def __init__(self, in_channels_edges, in_channels_nodes, hidden_channels, hidden_layers,
                 out_channels, kappa=3, p_dropout=0.0):
        super().__init__()
        self.in_channels_edges = in_channels_edges
        self.in_channels_nodes = in_channels_nodes
        self.hidden_channels = hidden_channels
        self.hidden_layers = hidden_layers
        self.out_channels = out_channels
        self.dense_channels = [hidden_channels*2, hidden_channels, out_channels]
        self.kappa = kappa
        self.p_dropout = p_dropout
        self.act = nn.ELU() 

    def reset_parameters(self):
        raise NotImplementedError

    def forward(self, data):
        raise NotImplementedError
        
    def aggr_x(self, x_dict):
        return x_dict['z_low'] + x_dict['z_up'] + x_dict['z_har']


class SCN(BaseNet):
    def __init__(self, in_channels_edges, in_channels_nodes, hidden_channels, hidden_layers, out_channels, kappa=3,
                 p_dropout=0.0, pooling_type=None, pooling_ratio=None, aggregate=True):
        super().__init__(in_channels_edges, in_channels_nodes, hidden_channels, hidden_layers,
                         out_channels, kappa, p_dropout)
        self.pooling_type = pooling_type
        self.convs = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.pool = nn.ModuleList([])

        self.convs.append(SCLayer(in_channels_edges, hidden_channels, kappa))
        self.pool.append(get_pooling_fn(pooling_type, pooling_ratio, hidden_channels, aggregate))
        self.bn.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(hidden_layers - 1):
            self.convs.append(SCLayer(hidden_channels, hidden_channels, kappa))
            self.pool.append(get_pooling_fn(pooling_type, pooling_ratio, hidden_channels, aggregate))
            self.bn.append(nn.BatchNorm1d(hidden_channels))
        self.out_dense = MLP(self.dense_channels, dropout=p_dropout)

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bn:
            bn.reset_parameters()
        for pl in self.pool:
            pl.reset_parameters()
        self.out_dense.reset_parameters()

    def forward(self, cochain):
        x, lower_index, upper_index, lower_values, upper_values = cochain.unpack_cochain()
        batch = cochain.x_edge_batch

        for i in range(self.hidden_layers):
            x = self.convs[i](x, lower_index, upper_index, lower_values, upper_values)  
            if self.pooling_type != 'sep_topk':
                x = self.aggr_x(x)
                x = self.bn[i](x)
                x = self.act(x)
            x, batch, lower_index, upper_index, lower_values, upper_values, _ = self.pool[i](
                x, batch, lower_index, upper_index, lower_values, upper_values)
        
        x = self.act(torch.cat([gmp(x, batch), gap(x, batch)], dim=1))
        return self.out_dense(x)



class HierSCN(SCN):
    def __init__(self, in_channels_edges, in_channels_nodes, hidden_channels, hidden_layers, out_channels, kappa=3,
                 p_dropout=0.0, pooling_type=None, pooling_ratio=None, aggregate=True):
        super().__init__(in_channels_edges, in_channels_nodes, hidden_channels, hidden_layers, out_channels, kappa,
                         p_dropout, pooling_type, pooling_ratio, aggregate)

    def forward(self, cochain):
        x, lower_index, upper_index, lower_values, upper_values = cochain.unpack_cochain()
        batch = cochain.x_edge_batch

        x  = self.convs[0](x, lower_index, upper_index, lower_values, upper_values)
        if self.pooling_type != 'sep_topk':
            x = self.aggr_x(x)
            x = self.bn[0](x)
            x = self.act(x)
        x, batch, lower_index, upper_index, lower_values, upper_values, _ = self.pool[0](
            x, batch, lower_index, upper_index, lower_values, upper_values)
        res = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        for i in range(1, self.hidden_layers):
            x = self.convs[i](x, lower_index, upper_index, lower_values, upper_values)
            if self.pooling_type != 'sep_topk':
                x = self.aggr_x(x)
                x = self.bn[i](x)
                x = self.act(x)
            x, batch, lower_index, upper_index, lower_values, upper_values, _ = self.pool[i](
                x, batch, lower_index, upper_index, lower_values, upper_values)
            res += torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.act(res)
        return self.out_dense(x)


class HierGCN(BaseNet):
    def __init__(self, in_channels_edges, in_channels_nodes, hidden_channels, hidden_layers, out_channels, kappa=3,
                 p_dropout=0.0, pooling_type=None, pooling_ratio=None, aggregate=True):
        super().__init__(in_channels_edges, in_channels_nodes, hidden_channels, hidden_layers,
                         out_channels, kappa, p_dropout)

        self.convs = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.pool = nn.ModuleList([])

        self.convs.append(GCNConv(in_channels_nodes, hidden_channels))
        self.pool.append(get_pooling_fn(pooling_type, pooling_ratio, hidden_channels, aggregate))
        self.bn.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(hidden_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.pool.append(get_pooling_fn(pooling_type, pooling_ratio, hidden_channels, aggregate))
            self.bn.append(nn.BatchNorm1d(hidden_channels))
        self.out_dense = MLP(self.dense_channels, dropout=p_dropout)

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bn:
            bn.reset_parameters()
        for pl in self.pool:
            pl.reset_parameters()
        self.out_dense.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
    
        x = self.act(self.convs[0](x, edge_index))
        x, edge_index, _, batch, perm, score = self.pool[0](x, edge_index, None, batch)
        res = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        for i in range(1, self.hidden_layers):
            x = self.act(self.convs[i](x, edge_index))
            x, edge_index, _, batch, _, score = self.pool[i](x, edge_index, None, batch)
            res += torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.act(res)
        return self.out_dense(x)
