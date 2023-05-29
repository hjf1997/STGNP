from data.base_dataset import BaseDataset
import torch
import pandas as pd
from math import radians, cos, sin, asin, sqrt
import numpy as np
import datetime
import random


class BJAirDEPDataset(BaseDataset):
    """
    Note that the beijing air quality dataset contains a lot of missing values, we need to handle this explicitly.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        parser.set_defaults(y_dim=1, covariate_dim=30)
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        """
        load data give options
        """
        self.opt = opt

        self.pred_attrs = [opt.pred_attr]
        self.drop_attrs = ['PM25_Concentration', 'PM10_Concentration','NO2_Concentration', 'CO_Concentration', 'O3_Concentration', 'SO2_Concentration']
        self.drop_attrs.remove(opt.pred_attr)
        self.drop_attrs += ['PM25_Missing','PM10_Missing', 'NO2_Missing', 'CO_Missing', 'O3_Missing', 'SO2_Missing']

        context_location_path = 'dataset/bjair/station.csv'
        target_location_path = 'dataset/bjair/Dense_Visualization/geo_locations.csv'

        test_context_path = 'dataset/bjair/Dense_Visualization/test_context.csv'
        test_target_path = 'dataset/bjair/Dense_Visualization/test_target.csv'

        print('Loading air quality features...')
        self.beijing_aq_context, context_stations_id, self.beijing_aq_target, target_stations_id = self.load_feat(test_context_path, test_target_path, opt.delete_col)

        print('Loading station locations...')
        self.A = self.load_loc(context_location_path, context_stations_id, target_location_path)

    def __len__(self):
        """Return the total number of samples in the dataset."""
        length = int(len(self.beijing_aq_context[list(self.beijing_aq_context.keys())[0]]['feat']) / self.opt.t_len)
        return length

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        """
        index_start = index * self.opt.t_len
        index_end = index * self.opt.t_len + self.opt.t_len

        x_target, x_context, y_context, missing_index_context = [], [], [], []

        for target in self.beijing_aq_target.keys():
            _x_target = self.beijing_aq_target[target]['feat'][index_start:index_end]
            x_target.append(torch.from_numpy(_x_target.to_numpy()).unsqueeze(0))
        x_target = torch.cat(x_target, dim=0)

        for context in self.beijing_aq_context.keys():
            _x_context = self.beijing_aq_context[context]['feat'][index_start:index_end]
            _missing = self.beijing_aq_context[context]['missing'][index_start:index_end]
            missing_index_context.append(torch.from_numpy(_missing.to_numpy()).squeeze(-1).unsqueeze(0))
            x_context.append(torch.from_numpy(_x_context.to_numpy()).unsqueeze(0))

            _y_context = self.beijing_aq_context[context]['pred'][index_start:index_end]
            y_context.append(torch.from_numpy(_y_context.to_numpy()).unsqueeze(0))
        x_context, y_context = torch.cat(x_context, dim=0), torch.cat(y_context, dim=0)

        time = self.beijing_aq_context[context]['feat'][index_start:index_end].index.values.astype(np.datetime64)
        time = (time - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')

        missing_index_context = torch.cat(missing_index_context, dim=0)
        t2c_A = self.A

        # different models have different settings for adjacency
        norm_A = np.exp(- 0.5 * (t2c_A / np.std(t2c_A)) ** 2)
        adj = torch.from_numpy(norm_A).unsqueeze(0).repeat(2, 1, 1)
        return {'feat_context': x_context.float(),  # [num_n, time, d_x]
                'pred_context': y_context.float(),  # [num_n, time, d_y]

                'feat_target': x_target.float(),  # [num_m, time, d_x]
                'pred_target': y_context.float(),  # [num_m, time, d_y] useless in this class

                'adj': adj.float(),  # [2, num_m, num_n]

                'missing_mask_context': missing_index_context.float(),  # [num_n, time]
                'missing_mask_target': missing_index_context.float(),  # [num_m, time] useless in this class

                'time': time  # [time]
        }

    def load_loc(self, context_location_path, context_stations_id, target_location_path, is_adj=True):
        # load air quality station locations data
        context_location = pd.read_csv(context_location_path)
        target_location = pd.read_csv(target_location_path)

        # context station locations for adj construction
        context_location = context_location.drop(columns=['name_english', 'name_chinese', 'district_id'])
        context_location = context_location[
            context_location['station_id'].isin(context_stations_id)].sort_values(by=['station_id']).reset_index(drop=True)
        # load target station locations for adj construction
        target_location = target_location.drop(columns=['district_id'])
        target_location = target_location.sort_values(by=['station_id'])

        if is_adj:
            # build adjacency matrix for each target node
            A = np.zeros((len(target_location), len(context_location)))
            for t in range(len(target_location)):
                for c in range(len(context_location)):
                    dis = self.haversine(target_location.at[t, 'longitude'],
                                                target_location.at[t, 'latitude'],
                                                context_location.at[c, 'longitude'],
                                                context_location.at[c, 'latitude'])
                    A[t, c] = dis
        else:
            A = np.zeros((len(target_location), len(context_location), 2))
            for t in range(len(target_location)):
                for c in range(len(context_location)):
                    A[t, c, 0] = target_location.at[t, 'longitude'] - context_location.at[c, 'longitude']
                    A[t, c, 1] = target_location.at[t, 'latitude'] - context_location.at[c, 'latitude']
        return A

    def load_feat(self, context_path, target_path, delete_col=None):

        # process context set
        beijing_multimodal = pd.read_csv(context_path, header=0)
        context_stations_id = beijing_multimodal.station_id.unique().tolist()
        # group by station beijing station id 1001-1036
        beijing_context = {}
        for id, station_aq in beijing_multimodal.groupby('station_id'):
            station_aq = station_aq.set_index("time").drop(columns=['station_id'])
            if delete_col is not None:
                station_aq = station_aq.drop(columns=delete_col)
            # split data into features and labels
            beijing_context[id] = {}
            beijing_context[id]['feat'] = station_aq.drop(columns=self.pred_attrs+self.drop_attrs)
            beijing_context[id]['missing'] = station_aq[[attr.split('_')[0]+'_Missing' for attr in self.pred_attrs]]
            beijing_context[id]['pred'] = station_aq[self.pred_attrs]

        # process target set
        beijing_multimodal = pd.read_csv(target_path, header=0)
        target_stations_id = beijing_multimodal.station_id.unique().tolist()
        beijing_target = {}
        for id, station_aq in beijing_multimodal.groupby('station_id'):
            station_aq = station_aq.set_index("time").drop(columns=['station_id'])
            if delete_col is not None:
                station_aq = station_aq.drop(columns=delete_col)
            beijing_target[id] = {}
            beijing_target[id]['feat'] = station_aq

        return beijing_context, context_stations_id, beijing_target, target_stations_id

