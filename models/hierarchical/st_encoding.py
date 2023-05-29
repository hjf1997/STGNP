import torch.nn as nn
import torch


#####################################
# Networks for deterministic path
#####################################
class Deterministic(nn.Module):
    def __init__(self, input_dim, covariate_dim, emd_channel, tcn_channels, tcn_kernel_size, dropout=0.1):
        """
        Args:
            tcn_channels: list
            tcn_kernel_size: int
            dropout: float
        """
        super().__init__()

        self.channels = tcn_channels
        side_channels = [covariate_dim] + tcn_channels
        tcn_channels = [emd_channel] + tcn_channels

        self.feature_embedding = Conv1d(input_dim, emd_channel, 1, actv=False)
        # self.aggr_encoding = nn.ModuleList([nn.Conv1d(tcn_channels[i], tcn_channels[i+1], 1) for i in range(len(tcn_channels)-1)])
        # todo: hard coding part, need to be changed
        self.graph_aggregator = nn.ModuleList([GraphAggregation(tcn_channels[i], tcn_channels[i], order=2) for i in range(len(tcn_channels)-1)])
        self.tcn = nn.ModuleList([TConv1d(tcn_channels[i], tcn_channels[i+1], tcn_kernel_size, 2 ** i, dropout) for i in range(len(tcn_channels)-1)])

        if covariate_dim > 0:
            self.side_encoding = nn.ModuleList([Conv1d(side_channels[i], side_channels[i+1], 1, dropout=dropout) for i in range(len(side_channels) - 1)])
        # self.output_projection = nn.ModuleList([Conv1d(tcn_channels[i], tcn_channels[i], 1, dropout=dropout) for i in range(1, len(tcn_channels))])

        # residual connection
        self.residual = nn.ModuleList([Conv1d(tcn_channels[i], tcn_channels[i+1], 1, actv=False) for i in range(len(tcn_channels)-1)])

        self.empty_token = nn.Parameter(torch.zeros([1, 1, emd_channel, 1]))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_context, y_context, x_target, y_target, adj, missing_mask_context):
        """
        Deterministic path: return all intermediate states, bottom-up
        Args:
            x_context: [batch, num_n, dx, time] (this strange shape is due to the requirement of Conv1d )
            y_context: [batch, num_n, dy, time]
            x_target: [batch, num_m, dx, time] (None if dim_feat is 0)
            y_target: [batch, num_m, dy, time] (None if generative, else inference)
            adj: adjacency matrix for target_node [batch, k_hop, num_m, num_n]
            missing_mask_context: missing nodes (1: missing) [batch, time, num_n]
        Returns:
            intermediate states list(tensor([batch * num_m, d, time]) * num_layers)
        """
        b, num_n, dy, t = y_context.shape
        num_m = adj.shape[2]
        target, context = y_target, y_context
        # feature embedding
        if target is None:
            target = self.empty_token.repeat([b, num_m, 1, t])
        else:
            target = self.feature_embedding(target)
        context = self.feature_embedding(context)

        d_t = []
        d_c = []
        for i in range(len(self.channels) + 1 - 1):
            target_resi = target.clone()
            context_resi = context.clone()

            # context = self.aggr_encoding[i](context.view([b*num_n, -1, t])).view([b, num_n, -1, t])
            target = self.graph_aggregator[i](context, target, adj, missing_mask_context)
            # context, target = torch.relu(self.dropout(context)), torch.relu(self.dropout(target))

            # temporal convolution
            target = self.tcn[i](target)
            context = self.tcn[i](context)

            # side information
            if x_target is not None:
                x_target = self.side_encoding[i](x_target)
                target = target + x_target
                x_context = self.side_encoding[i](x_context)
                context = context + x_context

            # residual connection
            if i > 0:
                target = torch.relu(target + self.residual[i](target_resi))
                context = torch.relu(context + self.residual[i](context_resi))

            d_t += [target]
            d_c += [context]
        return d_c, d_t


class GraphAggregation(nn.Module):

    def __init__(self, c_in, c_out, order, dropout=0.1):
        super().__init__()
        c_in = (order + 1) * c_in  # 1 for residual connection
        self.mlp = Conv1d(c_in, c_out, 1, dropout=dropout, actv=True)
        self.order = order

    def forward(self, feat_context, feat_target, adj, missing_mask_context):
        """
        Cross set graph neural network
        m: target set
        n: context set
        Args:
            feat_context: [batch, num_n, d, time]
            feat_target: [batch, num_m, d, time]
            adj: adjacency matrix for target_node [batch, k_hop, num_m, num_n]
            missing_mask_context: index the missing nodes (1: missing) [batch, time, num_n]
        Returns:
            feat_target: [batch, num_m, d_o, time]
        """
        out = [feat_target]
        for i in range(self.order):
            out += [self.aggregate(feat_context, feat_target, adj[:, i], missing_mask_context)]
        out = torch.cat(out, dim=2)
        feat_target = self.mlp(out)
        return feat_target

    def aggregate(self, feat_context, feat_target, adj, missing_index_context):
        feat_context, feat_target = feat_context.permute(0, 3, 1, 2), feat_target.permute(0, 3, 1, 2)  # [batch, time, num_n, d]
        b, t, num_n = feat_context.shape[:3]
        num_m = feat_target.shape[2]
        adj = torch.tile(adj.unsqueeze(1), [1, t, 1, 1])
        missing_index_context = torch.tile(missing_index_context.unsqueeze(2), [1, 1, num_m, 1])
        adj = adj * (1 - missing_index_context)
        norm_adj = adj / (torch.sum(adj, dim=-1, keepdim=True) + 1)  # 1 for the target node
        feat_target = (feat_target + torch.sum(feat_context.unsqueeze(2) * norm_adj.unsqueeze(-1), dim=-2))
        feat_target = feat_target.permute(0, 2, 3, 1)
        return feat_target

class TConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super(TConv1d, self).__init__()

        self.padding = nn.ConstantPad1d(((kernel_size - 1) * dilation, 0), 0)
        self.convolution = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if len(x.shape) == 4:
            b, n = x.shape[:2]
            x = x.flatten(0, 1)
            y = self.convolution(self.padding(x))
            y = y.reshape([b, n, -1, y.shape[-1]])
        else:
            y = self.convolution(self.padding(x))
        y = self.dropout(torch.relu(y))
        return y


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dropout=0.1, actv=True):
        super(Conv1d, self).__init__()
        self.convolution = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.dropout = nn.Dropout(dropout)
        self.actv = actv

    def forward(self, x):
        if len(x.shape) == 4:
            b, n = x.shape[:2]
            x = x.flatten(0, 1)
            y = self.convolution(x)
            y = y.reshape([b, n, -1, y.shape[-1]])
        else:
            y = self.convolution(x)
        if self.actv:
            y =  torch.relu(self.dropout(y))
        return y