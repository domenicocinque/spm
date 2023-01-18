from src.models.components.conv_layers import *
from src.models.components.nn_utils import *
from torch_geometric.nn import TopKPooling as GTopK, SAGPooling as GSAG


class BasePooling(nn.Module):
    def __init__(self, ratio=0.5, aggregate=True, aggregation_type='max'):
        super().__init__()
        self.ratio = ratio
        self.aggregate = aggregate
        self.aggregation_type = aggregation_type

    def forward(self, x_dict, batch, lower_index, upper_index, lower_values, upper_values):
        raise NotImplementedError

    def reset_parameters(self):
        pass

    @staticmethod
    def filter_both_adj(lower_index, upper_index, lower_values, upper_values, sampling_set, size):
        lower_index, lower_values = filter_adj(lower_index, lower_values, sampling_set, size)
        upper_index, upper_values = filter_adj(upper_index, upper_values, sampling_set, size)
        return lower_index, upper_index, lower_values, upper_values

    def __repr__(self):
        return f'{self.__class__.__name__}({self.ratio})'


class NaivePooling(BasePooling):
    def __init__(self, ratio, pooling_type, aggregate=True, aggregation_type='max'):
        super().__init__(ratio, aggregate, aggregation_type)
        self.pooling_type = pooling_type

    def forward(self, x, batch, lower_index, upper_index, lower_values, upper_values):
        size = x.size(0)
        if self.aggregate:
            x = pool_neighbor_x(x, lower_index, upper_index, self.aggregation_type)

        if self.pooling_type == 'random':
            xs = torch.rand((x.shape[0],), device=x.device)
        elif self.pooling_type == 'max':
            xs = x.sum(1).abs()

        sampling_set = topk(xs, self.ratio, batch)
        x = x[sampling_set]
        batch = batch[sampling_set]

        lower_index, upper_index, lower_values, upper_values = self.filter_both_adj(
            lower_index, upper_index, lower_values, upper_values, sampling_set, size)
        return x, batch, lower_index, upper_index, lower_values, upper_values, sampling_set


class TopKPooling(BasePooling):
    """
    Adapted from PyG
    """
    def __init__(self, in_channels, ratio, aggregate=True, aggregation_type='max', nonlinearity=torch.tanh):
        super().__init__(ratio, aggregate, aggregation_type)
        self.nonlinearity = nonlinearity
        self.weight = torch.nn.Parameter(torch.Tensor(1, in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.weight)

    def forward(self, x, batch, lower_index, upper_index, lower_values, upper_values):
        size = x.size(0)
        if self.aggregate:
            x = pool_neighbor_x(x, lower_index, upper_index, self.aggregation_type)

        score = (x * self.weight).sum(dim=-1)
        score = self.nonlinearity(score / self.weight.norm(p=2, dim=-1))
        sampling_set = topk(score, self.ratio, batch)

        x = x[sampling_set] * score[sampling_set].view(-1, 1)
        batch = batch[sampling_set]

        lower_index, upper_index, lower_values, upper_values = self.filter_both_adj(
            lower_index, upper_index, lower_values, upper_values, sampling_set, size)
        return x, batch, lower_index, upper_index, lower_values, upper_values, sampling_set


class SeparatedTopKPooling(BasePooling):
    def __init__(self, in_channels, ratio, aggregate=True, aggregation_type='max', nonlinearity=torch.tanh):
        super().__init__(ratio, aggregate, aggregation_type)
        self.nonlinearity = nonlinearity
        self.weight_low = torch.nn.Parameter(torch.Tensor(1, in_channels))
        self.weight_up = torch.nn.Parameter(torch.Tensor(1, in_channels))
        self.weight_har = torch.nn.Parameter(torch.Tensor(1, in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.weight_low)
        torch.nn.init.uniform_(self.weight_up)
        torch.nn.init.uniform_(self.weight_har)

    def forward(self, x_dict, batch, lower_index, upper_index, lower_values, upper_values):
        x_low, x_up, x_har = x_dict['z_low'], x_dict['z_up'], x_dict['z_har']
        size = x_low.size(0)
        if self.aggregate:
            x_low = pool_neighbor_x(x_low, lower_index, None, self.aggregation_type)
            x_up = pool_neighbor_x(x_up, upper_index, None, self.aggregation_type)
            x_har = pool_neighbor_x(x_har, lower_index, upper_index, self.aggregation_type)
        score_low = (x_low * self.weight_low).sum(dim=-1)/self.weight_low.norm(p=2, dim=-1)
        score_up = (x_up * self.weight_up).sum(dim=-1)/self.weight_up.norm(p=2, dim=-1)
        score_har = (x_har * self.weight_har).sum(dim=-1)/ self.weight_har.norm(p=2, dim=-1)
        score = score_low + score_up + score_har
        assert score is not None

        sampling_set = topk(score, self.ratio, batch)

        x_low = x_low[sampling_set] * self.nonlinearity(score_low[sampling_set]).view(-1, 1)
        x_up = x_up[sampling_set] * self.nonlinearity(score_up[sampling_set]).view(-1, 1)
        x_har = x_har[sampling_set] * self.nonlinearity(score_har[sampling_set]).view(-1, 1)
        x = x_low + x_up + x_har
        batch = batch[sampling_set]
        lower_index, upper_index, lower_values, upper_values = self.filter_both_adj(
            lower_index, upper_index, lower_values, upper_values, sampling_set, size)
        return x, batch, lower_index, upper_index, lower_values, upper_values, sampling_set


class SAGPooling(BasePooling):
    """
    Adapted from PyG
    """
    def __init__(self, in_channels, ratio, aggregate=True, aggregation_type='max', nonlinearity=torch.tanh):
        super().__init__(ratio, aggregate, aggregation_type)
        self.nonlinearity = nonlinearity
        self.scn = SCLayer(in_channels, 1, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.scn.reset_parameters()

    def forward(self, x, batch, lower_index, upper_index, lower_values, upper_values):
        size = x.size(0)
        if self.aggregate:
            x = pool_neighbor_x(x, lower_index, upper_index, self.aggregation_type)

        score = self.scn(x, lower_index, upper_index, lower_values, upper_values, sum_components=True).view(-1)
        score = self.nonlinearity(score)
        sampling_set = topk(score, self.ratio, batch)

        x = x[sampling_set] * score[sampling_set].view(-1, 1)
        batch = batch[sampling_set]

        lower_index, upper_index, lower_values, upper_values = self.filter_both_adj(
            lower_index, upper_index, lower_values, upper_values, sampling_set, size)
        return x, batch, lower_index, upper_index, lower_values, upper_values, sampling_set


class Test(BasePooling):
    pass


class NoPool(BasePooling):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, batch, lower_index, upper_index, lower_values, upper_values):
        return x, batch, lower_index, upper_index, lower_values, upper_values, None


def get_pooling_fn(pooling_type, ratio, in_channels=None, aggregate=True):
    if pooling_type in ['max', 'random']:
        return NaivePooling(ratio, pooling_type, aggregate=aggregate)
    elif pooling_type == 'topk':
        return TopKPooling(in_channels, ratio, aggregate=aggregate)
    elif pooling_type == 'sag':
        return SAGPooling(in_channels, ratio, aggregate=aggregate)
    elif pooling_type == 'sep_topk':
        return SeparatedTopKPooling(in_channels, ratio, aggregate=aggregate)
    elif pooling_type == 'graph_topk':
        return GTopK(in_channels, ratio)
    elif pooling_type == 'graph_sag':
        return GSAG(in_channels, ratio)
    elif pooling_type == 'test':
        return Test(in_channels, ratio, aggregate=aggregate)
    elif pooling_type is None or pooling_type == 'none':
        return NoPool()
    else:
        raise ValueError('Unknown pooling type: {}'.format(pooling_type))
