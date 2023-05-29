from models import init_net, BaseModel
import torch.nn as nn
import torch
from torch.distributions.kl import kl_divergence

from models.hierarchical.inference_model import InferenceModel
from models.hierarchical.likelihood_model import ObservationModel
from models.hierarchical.st_encoding import Deterministic
from utils.util import _mae_with_missing,_rmse_with_missing, _mape_with_missing


class HierarchicalModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        # modify options for the model
        parser.add_argument('--delete_col', default=['u_speed', 'v_speed', 'latitude', 'longitude']
                            , help='HNP does not use these attributes')
        parser.add_argument('--use_adj', default=True)
        return parser

    def __init__(self, opt, model_config):
        super().__init__(opt, model_config)
        """
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['nll', 'kl']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['HierarchicalNP']

        # specify metrics you want to evaluate the model. The training/test scripts will call functions in order:
        # <BaseModel.compute_metrics> compute metrics for current batch
        # <BaseModel.get_current_metrics> compute and return mean of metrics, clear evaluation cache for next evaluation
        self.metric_names = ['MAE', 'RMSE', 'MAPE']

        # define networks. The model variable name should begin with 'self.net'
        model_config['input_dim'] = opt.y_dim
        model_config['covariate_dim'] = opt.covariate_dim
        self.netHierarchicalNP = HierarchicalNP(model_config)
        self.netHierarchicalNP = init_net(self.netHierarchicalNP, opt.init_type, opt.init_gain, opt.gpu_ids)  # initialize parameters, move to cuda if applicable

        # define loss functions
        if self.isTrain:
            self.criterion = self.loss
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = torch.optim.Adam(self.netHierarchicalNP.parameters(), lr=opt.lr, betas=(0.9, 0.999))
            self.optimizers.append(self.optimizer)

        # results cache
        self.results = {}
        # KL flag, this is because some initializations are bad (kl will always be 0, terminate the program)
        self.kl_flag = False

    def set_input(self, input):
        """
        parse input for one epoch. data should be stored as self.xxx which would be adopted in self.forward().
        :param input: dict
        :return: None
        """
        self.feat_context = input['feat_context'].transpose(2, 3).to(self.device) if 'feat_context' in input.keys() else None
        self.pred_context = input['pred_context'].transpose(2, 3).to(self.device) # [batch, num_n, d_y, time]
        self.feat_target = input['feat_target'].transpose(2, 3).to(self.device) if 'feat_target' in input.keys() else None
        self.pred_target = input['pred_target'].transpose(2, 3).to(self.device)

        self.adj = input['adj'].to(self.device)
        self.missing_mask_context = input['missing_mask_context'].transpose(1, 2).to(self.device)
        self.missing_mask_target = input['missing_mask_target'].transpose(1, 2).to(self.device)
        self.bach_time = input['time']

    def forward(self, training=True):
        self.p_y_pred, self.q_dists, self.p_dists, self.var_c, self.var_t = \
            self.netHierarchicalNP(self.feat_context, self.pred_context, self.feat_target,
                                   self.pred_target if training else None, self.adj, self.missing_mask_context, training)

    def backward(self):
        # loss values expected to be displayed should begin with 'self.loss'
        self.loss_nll, self.loss_kl = self.criterion(self.p_y_pred, self.pred_target, self.q_dists, self.p_dists, self.missing_mask_target)
        self.loss_kl = self.opt.beta * self.loss_kl
        loss_ = self.loss_nll + self.loss_kl
        loss_.backward()
        if self.loss_kl > 1e-3:
            self.kl_flag = True

    def cache_results(self):
        self._add_to_cache('missing_target', self.missing_mask_target.reshape([-1, self.missing_mask_target.shape[2]]))
        self._add_to_cache('missing_context', self.missing_mask_context.reshape([-1, self.missing_mask_context.shape[2]]))

        self._add_to_cache('y_target', self.pred_target.permute(0, 3, 1, 2).flatten(0, 1), reverse_norm=True)  # [time, num_m, dy]
        self._add_to_cache('y_context', self.pred_context.permute(0, 3, 1, 2).flatten(0, 1), reverse_norm=True)  # [time, num_n, dy]
        self._add_to_cache('y_pred', self.p_y_pred.mean.permute(0, 3, 1, 2).flatten(0, 1), reverse_norm=True)  # [time, num_m, dy]
        self._add_to_cache('variance', self.p_y_pred.variance.permute(0, 3, 1, 2).flatten(0, 1), reverse_varnorm=True)  # [time, num_m, dy]

        self._add_to_cache('time', self.bach_time.reshape([-1]))

    def compute_metrics(self):
        y_pred = self.results['y_pred']
        y_target = self.results['y_target']
        missing_index_target = self.results['missing_target']

        self.metric_MAE = _mae_with_missing(y_pred, y_target, missing_index_target)
        self.metric_RMSE = _rmse_with_missing(y_pred, y_target, missing_index_target)
        self.metric_MAPE = _mape_with_missing(y_pred, y_target, missing_index_target)

    def loss(self, p_y_pred, y_target, q_dists, p_dists, missing_index_target):
        """
        calculate log_likelihood and kl_divergence
        Args:
            p_y_pred: distributions of predictions [batch, num_m, dy, time]
            y_target: ground truth [batch, num_m, dy, time]
            q_dists: distributions of posterior variables list ([batch, num_m, dz, time] * num_layers)
            p_dists: distributions of prior variables list ([batch, num_m, dz, time] * num_layers)
            missing_index_target: index the missing nodes (1: missing) [batch, time, num_m]
        Returns:
            log_likelihood: float
            kl_divergence:  float
        """
        assert p_y_pred.mean.shape == y_target.shape
        # mean for 1-st, 2-nd dimension, sum for 3-th, 4-th dimension
        valid_index_target = 1 - missing_index_target.permute([0, 2, 1]).unsqueeze(2)
        time = valid_index_target.shape[-1]
        # rescale the loss at time dimension, as the time dimension use sum
        time_rescale = time / torch.sum(valid_index_target, dim=-1, keepdim=True)

        log_likelihood = p_y_pred.log_prob(y_target) * valid_index_target
        log_likelihood = log_likelihood / time_rescale
        log_likelihood = log_likelihood.mean(dim=[0, 1]).sum()

        kl = kl_divergence(q_dists[0], p_dists[0]) * valid_index_target
        kl = kl / time_rescale
        kl = kl.mean(dim=[0, 1]).sum()
        for i in range(1, len(p_dists)):
            kl_ = kl_divergence(q_dists[i], p_dists[i]) * valid_index_target
            kl_ = kl_ / time_rescale
            kl += kl_.mean(dim=[0, 1]).sum()
        return -log_likelihood, kl

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netHierarchicalNP, True)
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()


class HierarchicalNP(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        # deterministic path
        self.deter = Deterministic(model_config['input_dim'], model_config['covariate_dim'], model_config['emd_channel'], model_config['tcn_channels'], model_config['tcn_kernel_size'], model_config['dropout'])
        # stochastic path
        self.inference = InferenceModel(model_config['tcn_channels'], model_config['latent_channels'], model_config['num_latent_layers'])
        # observation path
        self.observation = ObservationModel(sum(model_config['latent_channels']),
                                            model_config['input_dim'], model_config['observation_hidden_dim'], model_config['num_observation_layers'])

    def forward(self, x_context, y_context, x_target, y_target, adj, missing_index_context, training):
        # generative model (conditional prior)
            # the name posterior is weird for context set. This is because we have y for the set, so that we can use this 'posterior encoder'.
            # However, it is still conditional prior theoretically in this sense
        p_d_c, p_d_t = self.deter(x_context, y_context, x_target, None, adj, missing_index_context)
        p_context, p_dists, var_c_cache, var_t_cache = self.inference(p_d_c, p_d_t, adj[:, 0], missing_index_context, training)

        if y_target is not None:
            # inference model (posterior)
            d_c, d_t = self.deter(x_context, y_context, x_target, y_target, adj, missing_index_context)
            q_target, q_dists, _, _ = self.inference(d_c, d_t, adj[:, 0], missing_index_context)

            # observation model (likelihood)
            p_y_pred = self.observation(q_target)
        else:
            # observation model (likelihood)
            p_y_pred = self.observation(p_context)
            q_dists = None

        return p_y_pred, q_dists, p_dists, var_c_cache, var_t_cache
