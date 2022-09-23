import torch
from torch import Tensor, nn, functional as F
from torch_geometric.nn import MLP
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros, glorot
from torch_geometric.typing import (
    Adj,
    OptTensor,
)
from torch_sparse import SparseTensor, matmul


class SMConv(MessagePassing):
    """
    Just a slight modification of TAGConv
    """
    def __init__(self, in_channels: int, out_channels: int, K: int = 3,
                 bias: bool = True, normalize: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalize = normalize

        self.lins = torch.nn.ModuleList([
            Linear(in_channels, out_channels, bias=False) for _ in range(K + 1)
        ])

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        if self.normalize:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim),
                    improved=False, add_self_loops=False, dtype=x.dtype)

            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim),
                    add_self_loops=False, dtype=x.dtype)

        out = self.lins[0](x)
        for lin in self.lins[1:]:
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                               size=None)
            out += lin.forward(x)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K})')


class SCLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, K=3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.K = K

        self.lower_conv = SMConv(in_channels, hidden_channels, K, normalize=False)
        self.upper_conv = SMConv(in_channels, hidden_channels, K, normalize=False)
        self.harmonic = nn.Linear(in_channels, hidden_channels, bias=False)

    def reset_parameters(self):
        self.lower_conv.reset_parameters()
        self.upper_conv.reset_parameters()
        self.harmonic.reset_parameters()

    def forward(self, x, lower_index, upper_index, lower_values=None, upper_values=None, sum_components=False):
        z_low = self.lower_conv(x, edge_index=lower_index, edge_weight=lower_values)
        z_up = self.upper_conv(x, edge_index=upper_index, edge_weight=upper_values)
        z_har = self.harmonic(x)

        if sum_components:
            return z_low + z_up + z_har
        else:
            return {'z_low': z_low, 'z_up': z_up, 'z_har': z_har}



