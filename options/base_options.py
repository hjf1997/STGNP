import argparse
import os
from utils import util
import torch
import models
import data
import time
import yaml
import numpy as np
import random
import sys

class BaseOptions():
    """This class defines options used during both training and test time.
    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # model parameters
        parser.add_argument('--model', type=str, default='', help='chose which model to use.')
        parser.add_argument('--config', type=str, default='config1', help='choose configurations for model')
        parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        # dataset parameters
        parser.add_argument('--pred_attr', type=str, default='PM25_Concentration', help='Which AQ attribute to infer')
        parser.add_argument('--dataset_mode', type=str, default='', help='chooses dataset')
        # extrapolation dataset path
        parser.add_argument('--t_len', type=int, default=24, help='time window for inference')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=0, type=int, help='# threads for loading data. Note: larger than 0 will throw out an error in my computer')
        parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
        parser.add_argument('--max_dataset_size', type=int, default=float('inf'), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--seed', type=int, default=2023, help='random seed for initialization')
        parser.add_argument('--enable_visual', action='store_true', help='enable visualization')
        # add you customized parameters below
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # load model configurations from .yaml
        yaml_path = os.path.join('model_configurations', opt.model + '_config.yaml')
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as config_file:
                model_config = yaml.safe_load(config_file)
                model_config = model_config[opt.config]
        else:
            raise FileNotFoundError('Cannot find configuration file.')

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser and model config
        self.parser = parser
        return parser.parse_args(), model_config

    def print_options(self, opt, model_config):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------\n'

        # print model configurations
        message += '----------------- Model Configurations ---------------\n'
        for k, v in model_config.items():
            message += '{:>25}: {:<30}\n'.format(str(k), str(v))
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

        # save system outputs to the file
        logger_file_name = os.path.join(expr_dir, '{}_error.log'.format(opt.phase))
        sys.stderr = Logger(filename=logger_file_name, stream=sys.stdout)

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt, model_config = self.gather_options()
        opt.isTrain = self.isTrain   # train or test
        opt.checkpoints_dir = os.path.join(opt.checkpoints_dir, opt.dataset_mode)
        if opt.phase == 'test' or opt.continue_train:
            if opt.file_time == '':
                raise RuntimeError('Please specify checkpoint time!')
            else:
                opt.name = opt.model + '_' + opt.pred_attr.replace('_Concentration', '') + '_' + opt.file_time
        else:
            current_time = time.strftime("%Y%m%dT%H%M%S", time.localtime())
            opt.name = opt.model + '_' + opt.pred_attr.replace('_Concentration', '') + '_' + current_time
            opt.file_time = current_time
        if opt.phase != 'val':
            self.print_options(opt, model_config)

        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.manual_seed_all(opt.seed)
            torch.cuda.set_device(opt.gpu_ids[0])
        self.opt = opt
        return self.opt, model_config


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.filename = filename

    def write(self, message):
        self.terminal.write(message)
        with open(self.filename, 'a') as log:
            log.write(message)

    def flush(self):
        pass