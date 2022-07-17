import networkx as nx
import torch
from torch.utils.data import random_split
import scipy.sparse.linalg as spl
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse

from src.datamodules.components.complex import CochainData


def get_data_loaders(data, batch_size, splits, num_workers):
    train_size = int(splits[0] * len(data))
    val_size = int(splits[1] * len(data))
    test_size = len(data) - train_size - val_size

    splits = random_split(data, [train_size, val_size, test_size])
    train_idx = splits[0].indices
    val_idx = splits[1].indices
    test_idx = splits[2].indices

    train_loader = DataLoader(data[train_idx], batch_size=batch_size, shuffle=True,
                              follow_batch=['x_edge'], num_workers=num_workers)
    val_loader = DataLoader(data[val_idx], batch_size=batch_size, follow_batch=['x_edge'],
                            num_workers=num_workers)
    if test_size > 0:
        test_loader = DataLoader(data[test_idx], batch_size=batch_size, follow_batch=['x_edge'],
                                 num_workers=num_workers)
    else:
        test_loader = None
    return train_loader, val_loader, test_loader


def sparse_lambda_max(index, values, dense_shape):
    A = torch.sparse_coo_tensor(indices=index, values=values, size=dense_shape)
    lambda_max = spl.eigsh(A.coalesce().to_dense().numpy(), k=1, which="LM", return_eigenvectors = False)[0]
    return torch.Tensor([lambda_max])


def normalize(data):
    E = data.x_edge.shape[0]
    lambda_low = sparse_lambda_max(data.lower_index, data.lower_values, (E, E))
    data.lower_values = data.lower_values / lambda_low
    if data.upper_values.shape[0] > 1:
        lambda_up = sparse_lambda_max(data.upper_index, data.upper_values, (E, E))
        data.upper_values = data.upper_values / lambda_up
    return data


# Partially adapted from https://github.com/ggoh29/Simplicial-neural-network-benchmark

def edge_to_node_matrix(edges, nodes, one_indexed=True):
    sigma1 = torch.zeros((len(nodes), len(edges)), dtype=torch.float)
    offset = int(one_indexed)
    j = 0
    for edge in edges:
        x, y = edge
        sigma1[x - offset][j] -= 1
        sigma1[y - offset][j] += 1
        j += 1
    return sigma1


def triangle_to_edge_matrix(triangles, edges):
    sigma2 = torch.zeros((len(edges), len(triangles)), dtype=torch.float)
    edges = [e for e in edges]
    edges = {edges[i]: i for i in range(len(edges))}
    for l in range(len(triangles)):
        i, j, k = triangles[l]
        if (i, j) in edges:
            sigma2[edges[(i, j)]][l] += 1
        else:
            sigma2[edges[(j, i)]][l] -= 1

        if (j, k) in edges:
            sigma2[edges[(j, k)]][l] += 1
        else:
            sigma2[edges[(k, j)]][l] -= 1

        if (i, k) in edges:
            sigma2[edges[(i, k)]][l] -= 1
        else:
            sigma2[edges[(k, i)]][l] += 1

    return sigma2


def build_upper_features(lower_features, simplex_list):
    new_features = []
    for i in range(len(simplex_list)):
        if isinstance(simplex_list[i], tuple):
            idx = list(simplex_list[i])
        else:
            idx = simplex_list[i]
        new_features.append(lower_features[idx].mean(axis=0))
    return torch.stack(new_features)


def get_incidences(nodes, edge_list):
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edge_list, color='white')
    triangle_list = [list(sorted(x)) for x in nx.enumerate_all_cliques(g) if len(x) == 3]
    lower_incidence = edge_to_node_matrix(list(edge_list), nodes, one_indexed=False)
    upper_incidence = triangle_to_edge_matrix(triangle_list, edge_list)
    return lower_incidence, upper_incidence


def convert_to_cochain(data):
    if data.x is None:
        features = torch.ones(data.edge_index.max().item() + 1, 1)
    elif data.x.size(1) == 0:
        features = torch.ones(data.size(0), 1)
    else:
        features = data.x
    edges, y = data.edge_index.reshape(-1, 2).tolist(), data.y
    nodes = torch.tensor([i for i in range(features.shape[0])])
    edge_list = [(i, j) for i, j in edges]

    # Compute simplexes
    lower_incidence, upper_incidence = get_incidences(nodes, edge_list)
    lower_laplacian = lower_incidence.T @ lower_incidence
    upper_laplacian = upper_incidence @ upper_incidence.T

    assert (upper_laplacian.shape == upper_laplacian.shape)

    x0 = features
    # Edge features
    x1 = build_upper_features(x0, edge_list)
    if data.edge_attr is not None:
        x1_add = data.edge_attr.float()
        if x1_add.dim() == 1:
            x1_add = x1_add.unsqueeze(1)
        x1 = torch.cat([x1, x1_add], dim=1)

    lower_index, lower_value = dense_to_sparse(lower_laplacian)
    upper_index, upper_value = dense_to_sparse(upper_laplacian)

    out = CochainData(lower_index, upper_index, lower_value, upper_value, x1, y.long())
    out.num_triangles = upper_incidence.shape[1]
    out.x = x0
    out.edge_index = data.edge_index
    return out
