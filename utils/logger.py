import os
import time
import yaml
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime

class Logger():
    """This class includes several functions that can display/save image data, loss values and print/save logging information.
    It depends on the online experiment tracking platform neptune.ai (https://neptune.ai/)
    """

    def __init__(self, opt):
        """Initialize the Visualizer class
        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.name = opt.name
        self.saved = False

        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        self.metrics_name = os.path.join(opt.checkpoints_dir, opt.name, 'metrics.txt')
        self.plot_dir = os.path.join(opt.checkpoints_dir, opt.name, 'plots')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)
        with open(self.metrics_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Metrics (%s) ================\n' % now)

        # neptune experiment tracking
        if opt.isTrain and opt.enable_neptune:
            import neptune.new as neptune
            try:
                self.neptune_run = neptune.init(project=opt.neptune_project,
                                   api_token=opt.neptune_token,
                                   source_files=['*.py'])
            except Exception as e:
                print(e)
                opt.enable_neptune = False
            self.neptune_options(opt)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk
        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.5f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
        if self.opt.phase != 'test' and self.opt.enable_neptune:
            self.neptune_current_losses(epoch, iters, losses, t_comp, t_data)

    def print_current_metrics(self, epoch, iters, metrics, t_val):
        """print current losses on console; also save the losses to the disk
        """
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, iters, t_val)
        for k, v in metrics.items():
            message += '%s: %.5f ' % (k, v)

        print(message)  # print the message
        with open(self.metrics_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
        if self.opt.phase != 'test' and self.opt.enable_neptune:
            self.neptune_current_metrics(epoch, iters, metrics, t_val)

    def save_visuals(self, visuals, phase, epoch, title=''):
        if phase == 'val': phase = 'validation'

        for k, v in visuals.items():
            visuals = v

        means = visuals['mean']
        variances = visuals['variance']
        y_target = visuals['y_target']
        time = visuals['time']
        time_str = [datetime.datetime.utcfromtimestamp(time[i]) for i in range(time.shape[0])]

        plot_dir = os.path.join(self.plot_dir, 'epoch_'+str(epoch))
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)
        for i in range(means.shape[1]):
            fig = plt.figure(figsize=(24, 6))
            plt.title(title, fontsize=14)
            plt.plot(time_str, y_target[:, i, 0], "x", label="Ground Truth", alpha=0.7, markersize=3)
            plt.plot(time_str, means[:, i, 0], "o", label="Predictions", alpha=0.7, markersize=3)
            (line,) = plt.plot(time_str, means[:, i, 0], lw=1., label="Mean of predictions")
            col = line.get_color()
            plt.fill_between(
                time_str,
                (means[:, i, 0] - 1 * variances[:, i, 0] ** 0.5),
                (means[:, i, 0] + 1 * variances[:, i, 0] ** 0.5),
                color=col,
                alpha=0.2,
                lw=0.5,
            )
            plt.legend()
            plt.grid()
            plt.ylim(bottom=0)
            plt.xlim(left=time_str[0] + datetime.timedelta(days=-1), right=time_str[-1] + datetime.timedelta(days=1))
            ax = plt.gca()
            date_format = mpl.dates.DateFormatter("%m-%d")
            ax.xaxis.set_major_formatter(date_format)  # 控制x轴显示日期的间隔天数（如一周7天）

            xlocator = mpl.ticker.MultipleLocator(1)
            ylocator = mpl.ticker.MultipleLocator(25)

            ax.xaxis.set_major_locator(xlocator)
            ax.yaxis.set_major_locator(ylocator)

            plt.xticks(rotation=45)

            plt.savefig(os.path.join(plot_dir, 'Node_'+str(i)+'.pdf'), dpi=600)
            if self.opt.phase != 'test' and self.opt.enable_neptune:
                self.neptune_run[phase + '/plot/' + 'epoch_' + str(epoch) + '/Node_' + str(i)].upload(fig)
            plt.close(fig)

    def neptune_options(self, opt):
        """
        print configurations to neptune
        :return:
        """
        # load model configuration from .yaml
        yaml_path = os.path.join('model_configurations', opt.model + '_config.yaml')
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as config_file:
                configs = yaml.safe_load(config_file)
                model_config = configs[opt.config]
            self.neptune_run['model_configs'] = model_config
        else:
            model_config = None

        config = {}
        for k, v in sorted(vars(opt).items()):
            if model_config and k in model_config.keys():
                continue
            config[k] = v
        self.neptune_run['framework_configs/'] = config

    def neptune_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses to neptune;
        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        for k, v in losses.items():
            self.neptune_run['train/'+k].log(v)
        self.neptune_run['train/computation time'].log(t_comp)
        self.neptune_run['train/data loading time'].log(t_data)

    def neptune_current_metrics(self, epoch, iters, metrics, t_val):
        """
        print metrics to neptune
        :param epoch:
        :param iters:
        :param metrics:
        :param t_val:
        :return:
        """
        for k, v in metrics.items():
            self.neptune_run['validation/'+k].log(v)
        self.neptune_run['validation/computation time'].log(t_val)

    def neptune_networks(self, model):
        """
        print the total number of parameter in the network to neptune
        :param model:
        :return:
        """
        for name in model.model_names:
            if isinstance(name, str):
                net = eval('model.net'+name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
            self.neptune_run['model/num_parameters/'+name] = num_params
