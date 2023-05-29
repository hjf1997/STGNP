import torch.nn as nn
import torch
from torch.distributions import Normal, LogNormal
import torch.nn.functional as F

########################################
# Observation model
########################################
class ObservationModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden_layers):
        super(ObservationModel, self).__init__()
        # dense observation model
        self.num_hidden_layers = num_hidden_layers

        self.dec_in = nn.Conv1d(input_dim, hidden_dim, 1)
        self.dec_hidden = nn.Sequential(*[nn.Conv1d(hidden_dim, hidden_dim, 1)
                                          for _ in range(num_hidden_layers)])
        self.dec_out_1 = nn.Conv1d(hidden_dim, output_dim, 1)
        self.dec_out_2 = nn.Conv1d(hidden_dim, output_dim, 1)

    def decode(self, z):
        b, num_m, _, t = z.shape
        z = z.view([b*num_m, -1, t])
        h = torch.relu(self.dec_in(z))
        for i in range(self.num_hidden_layers):
            h = torch.relu(self.dec_hidden[i](h))
        return self.dec_out_1(h).view([b, num_m, -1, t]), self.dec_out_2(h).view([b, num_m, -1, t])

    def forward(self, z):
        """
        likelihood function
        Args:
            z: list
        Returns:
            distributions of target_y
        """
        z = torch.cat(z, dim=2)
        mu, log_sigma = self.decode(z)
        return Normal(mu, 0.1 + 0.9 * F.softplus(log_sigma))