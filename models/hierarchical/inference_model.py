import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Normal, LogNormal


#####################################
# Networks for stochastic path
#####################################
class LatentLayer(nn.Module):
    def __init__(self, tcn_dim, latent_dim_in, latent_dim_out, hidden_dim, num_hidden_layers):
        super(LatentLayer, self).__init__()

        self.num_hidden_layers = num_hidden_layers

        self.enc_in = nn.Conv1d(tcn_dim + latent_dim_in, hidden_dim, 1)
        self.enc_hidden = nn.ModuleList([nn.Conv1d(hidden_dim, hidden_dim, 1)
                                          for _ in range(num_hidden_layers)])
        self.enc_out_1 = nn.Conv1d(hidden_dim, latent_dim_out, 1)
        self.enc_out_2 = nn.Conv1d(hidden_dim, latent_dim_out, 1)

    def forward(self, x):
        b, num_m, _, t = x.shape
        x = x.view([b*num_m, -1, t])
        h = torch.relu(self.enc_in(x))
        for i in range(self.num_hidden_layers):
            h = torch.relu(self.enc_hidden[i](h))
        mu = self.enc_out_1(h).view([b, num_m, -1, t])
        sigma = 0.1 + 0.9 * F.softplus(self.enc_out_2(h).view([b, num_m, -1, t]))
        return mu, sigma * sigma


class InferenceModel(nn.Module):
    def __init__(self, tcn_channels, latent_channels, num_hidden_layers):
        super(InferenceModel, self).__init__()

        self.layers_c = [LatentLayer(tcn_channels[i], 0, latent_channels[i],
                                   latent_channels[i], num_hidden_layers) for i in range(len(tcn_channels))]
        self.layers_c = nn.ModuleList(self.layers_c)
        self.latent_layer = [nn.Conv1d(latent_channels[i], tcn_channels[i-1], 1) for i in range(1, len(tcn_channels))]
        self.latent_layer = nn.ModuleList(self.latent_layer)
        # self.layers_t = [LatentLayer(tcn_channels[i], latent_channels[i + 1], latent_channels[i],
        #                            latent_channels[i], num_hidden_layers) for i in range(len(tcn_channels) - 1)]
        # self.layers_t += [LatentLayer(tcn_channels[-1], 0, latent_channels[-1], latent_channels[-1],
        #                             num_hidden_layers)]
        # self.layers_t = nn.ModuleList(self.layers_t)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # 0.5 means sqrt
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, d_c, d_t, adj, missing_index_context, training=True):
        """

        Args:
            d_c: list[[batch, num_n, d, time], ...]
            d_t: list[[batch, num_m, d, time], ...]
            adj: adjacency matrix for target_node [batch, num_m, num_n]
            missing_index_context: index the missing nodes (1: missing) [batch, time, num_n]
            training: training flat
        Returns:

        """
        b, num_m, num_n = adj.shape
        t = d_c[0].shape[-1]

        adj = torch.tile(adj.unsqueeze(1), [1, t, 1, 1])
        missing_index_context = torch.tile(missing_index_context.unsqueeze(2), [1, 1, num_m, 1])
        adj = adj * (1 - missing_index_context)
        norm_adj = adj / (torch.sum(adj, dim=-1, keepdim=True) + 1)  # 1 for the target node
        norm_adj = norm_adj.permute(0, 2, 3, 1 )  # [batch, num_m, num_n, time]

        # variance cache
        var_c_cache = []
        var_t_cache = []

        # top-down
        r_c, var_c = self.layers_c[-1](d_c[-1])
        mu_t, var_t = self.layers_c[-1](d_t[-1])
        var_c_cache.append(var_c.detach()), var_t_cache.append(var_t.detach())

        # Gaussian conditioning
        var_t_agg = 1 / ((1 / var_t) + torch.sum(norm_adj.unsqueeze(-2) ** 2 / var_c.unsqueeze(1), dim=-3))
        mu_t_agg = var_t_agg * (
                        mu_t / var_t + torch.sum(norm_adj.unsqueeze(-2) ** 2 * r_c.unsqueeze(1) / var_c.unsqueeze(1), dim=2))
        dists = [Normal(mu_t_agg, torch.sqrt(var_t_agg))]
        z = [dists[-1].rsample()] if training else [dists[-1].mean]

        for i in reversed(range(len(self.layers_c)-1)):
            z_next = torch.relu(self.latent_layer[i](z[-1].view([b*num_m, -1, t]))).view([b, num_m, -1, t])
            mu_t, var_t = self.layers_c[i](d_t[i] + z_next)
            r_c, var_c = self.layers_c[i](d_c[i])
            var_c_cache.append(var_c.detach()), var_t_cache.append(var_t.detach())

            var_t_agg = 1 / ((1 / var_t) + torch.sum(norm_adj.unsqueeze(-2) ** 2 / var_c.unsqueeze(1), dim=-3))
            mu_t_agg = var_t_agg * (
                        mu_t / var_t + torch.sum(norm_adj.unsqueeze(-2) ** 2 * r_c.unsqueeze(1) / var_c.unsqueeze(1), dim=2))
            dists += [Normal(mu_t_agg, torch.sqrt(var_t_agg))]
            z += [dists[-1].rsample()] if training else [dists[-1].mean]

        return z, dists, var_c_cache, var_t_cache


class InferenceModelWithoutBayesianAggregation(nn.Module):
    def __init__(self, aggr_channels, latent_channels, num_hidden_layers):
        super(InferenceModelWithoutBayesianAggregation, self).__init__()

        self.layers = [LatentLayer(aggr_channels[i], latent_channels[i + 1], latent_channels[i],
                                   latent_channels[i], num_hidden_layers) for i in range(len(aggr_channels) - 1)]
        self.layers += [LatentLayer(aggr_channels[-1], 0, latent_channels[-1], latent_channels[-1],
                                    num_hidden_layers)]
        self.layers = nn.ModuleList(self.layers)

    def forward(self, d_c, d_t, adj, missing_index_context, training=True):
        """

        Args:
            d_c: [batch, num_n, d, time]
            d_t: [batch, num_m, d, time]
            adj: adjacency matrix for target_node [batch, num_m, num_n]
            missing_index_context: index the missing nodes (1: missing) [batch, time, num_n]
            training: training flat
        Returns:

        """
        # top-down
        mu, var = self.layers[-1](d_t[-1])
        # 0.1 + 0.9 * torch.sigmoid(self.hidden_to_sigma(hidden))
        dists = [Normal(mu, torch.sqrt(var))]
        z = [dists[-1].rsample()] if training else [dists[-1].mean]
        for i in reversed(range(len(self.layers)-1)):
            mu, var = self.layers[i](torch.cat((d_t[i], z[-1]), dim=2))
            dists += [Normal(mu, torch.sqrt(var))]
            z += [dists[-1].rsample()] if training else [dists[-1].mean]
        return z, dists
