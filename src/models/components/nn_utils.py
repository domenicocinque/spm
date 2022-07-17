import torch
from torch_geometric.utils import dense_to_sparse, add_self_loops, coalesce, to_dense_batch
from torch_scatter import scatter_add, scatter_max, scatter
import numpy as np
from scipy.sparse.linalg import eigsh


def sample_adjacency(adj_indices, adj_values, shape, idx):
    coo_adj = torch.sparse_coo_tensor(indices=adj_indices, values=adj_values, size=shape)
    sampled = coo_adj.coalesce().to_dense()[idx, :][:, idx]
    return dense_to_sparse(sampled)


def filter_adj(edge_index, edge_attr, perm, num_nodes=None):
    mask = perm.new_full((num_nodes, ), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i

    row, col = edge_index
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    return torch.stack([row, col], dim=0), edge_attr


def topk(x, ratio, batch, min_score=None, tol=1e-7):
    """
    From PyG
    """
    if min_score is not None:
        # Make sure that we do not drop all nodes in a graph.
        scores_max = scatter_max(x, batch)[0].index_select(0, batch) - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = (x > scores_min).nonzero(as_tuple=False).view(-1)
    else:
        num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
        batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

        cum_num_nodes = torch.cat(
            [num_nodes.new_zeros(1),
             num_nodes.cumsum(dim=0)[:-1]], dim=0)

        index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
        index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

        dense_x = x.new_full((batch_size * max_num_nodes,),
                             torch.finfo(x.dtype).min)
        dense_x[index] = x
        dense_x = dense_x.view(batch_size, max_num_nodes)

        _, perm = dense_x.sort(dim=-1, descending=True)
        perm = perm + cum_num_nodes.view(-1, 1)
        perm = perm.view(-1)

        if isinstance(ratio, int):
            k = num_nodes.new_full((num_nodes.size(0),), ratio)
            k = torch.min(k, num_nodes)
        else:
            k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)

        mask = [
            torch.arange(k[i], dtype=torch.long, device=x.device) +
            i * max_num_nodes for i in range(batch_size)
        ]
        mask = torch.cat(mask, dim=0)

        perm = perm[mask]

    return perm


def pool_neighbor_x(x, lower_index, upper_index=None, reduce='max'):
    """
    Adapted from PyG
    """
    if upper_index is None:
        adj_index = lower_index
    else:
        adj_index = coalesce(torch.cat([lower_index, upper_index], dim=1))
    adj_index, _ = add_self_loops(adj_index, num_nodes=x.shape[0])
    row, col = adj_index
    x = scatter(x[row], col, dim=0, dim_size=x.shape[0], reduce=reduce)
    return x


def get_sparse_hodge_laplacian(lower_index, upper_index, lower_values, upper_values, shape=None):
    lower_laplacian = torch.sparse_coo_tensor(indices=lower_index, values=lower_values, size=shape)
    upper_laplacian = torch.sparse_coo_tensor(indices=upper_index, values=upper_values, size=shape)
    hodge = (lower_laplacian + upper_laplacian).coalesce()
    return hodge.indices(), hodge.values()


def to_dense_adj(adj_indices, adj_values, shape):
    coo_adj = torch.sparse_coo_tensor(indices=adj_indices, values=adj_values, size=shape)
    return coo_adj.coalesce().to_dense()


def filter_batch_index(size, batch, ratio, sampling_set):
    """
    For each batch, take only num_nodes*ratio nodes.
    """
    num_nodes = scatter_add(batch.new_ones(size), batch, dim=0)
    batch_size = num_nodes.size(0)
    filt_nodes = (num_nodes * ratio).long()
    # To make sure that we do not drop all nodes in a graph.
    filt_nodes[filt_nodes == 0] = 1

    store = torch.ones(batch.size(0), dtype=torch.long, device=batch.device).neg()
    store[sampling_set] = sampling_set
    split, mask = to_dense_batch(store, batch)
    new_sampling_set = torch.cat(
        [split[i][split[i] >= 0][:filt_nodes[i]] for i in range(batch_size)])
    return new_sampling_set
