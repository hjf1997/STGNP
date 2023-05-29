from data.base_dataset import BaseDataset
import torch
import pandas as pd
from math import radians, cos, sin, asin, sqrt
import numpy as np
import datetime
import random


class LDAirDataset(BaseDataset):
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
        parser.set_defaults(dim_x=5, dim_y=1)
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        """
        load data give options
        """
        self.opt = opt

        self.pred_attrs = [opt.pred_attr]
        self.drop_attrs = ['PM25_Concentration', 'PM10_Concentration','NO2_Concentration']
        self.drop_attrs.remove(opt.pred_attr)

        self.drop_attrs += ['PM25_Missing','PM10_Missing', 'NO2_Missing']

        # aq_location_path = opt.aq_location_path
        aq_location_path = 'dataset/London/London_AirQuality_Stations.csv'

        train_path = opt.train_path
        test_context_path = opt.test_context_path
        test_target_path = opt.test_target_path

        print('Loading air quality features...')
        if opt.phase == 'train':
            self.beijing_aq, self.station_list = self.load_feat(train_path, opt.delete_col)
            self.station_list = sorted(self.station_list)
            # For adjacency matrix fetching
            self.station_index = [i for i in range(len(self.station_list))]
        else:
            beijing_aq_context, self.context_station_list = self.load_feat(test_context_path, opt.delete_col)
            self.context_station_list = sorted(self.context_station_list)
            beijing_aq_target, self.target_station_list = self.load_feat(test_target_path, opt.delete_col)
            self.target_station_list = sorted(self.target_station_list)
            self.station_list = sorted(self.context_station_list + self.target_station_list)
            self.beijing_aq = {**beijing_aq_context, **beijing_aq_target}

        print('Loading station locations...')
        if opt.phase == 'train':
            self.A = self.load_loc(aq_location_path, self.station_list, self.station_list, use_adj=opt.use_adj)
        else:
            self.A = self.load_loc(aq_location_path, self.context_station_list, self.target_station_list, use_adj=opt.use_adj)

    def __len__(self):
        """Return the total number of samples in the dataset."""
        if self.opt.phase == 'train':
            length = len(self.beijing_aq[self.station_list[0]]['feat']) - self.opt.t_len
        else:
            length = int(len(self.beijing_aq[self.station_list[0]]['feat']) / self.opt.t_len)
        return length

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        """
        if self.opt.phase == 'train':
            # random divide nodes into context set and target set
            target_station_index = random.sample(self.station_index, self.opt.num_train_target)
            self.target_station_list = np.array(self.station_list)[target_station_index].tolist()
            context_station_index = list(set(self.station_index).difference(set(target_station_index)))
            self.context_station_list = np.array(self.station_list)[context_station_index].tolist()

        if self.opt.phase == 'train':
            index_start = index
            index_end = index + self.opt.t_len
        else:
            index_start = index * self.opt.t_len
            index_end = index * self.opt.t_len + self.opt.t_len

        x_target, y_target, x_context, y_context, missing_index_context, missing_index_target = [], [], [], [], [], []

        for target in self.target_station_list:
            _x_target = self.beijing_aq[target]['feat'][index_start:index_end]
            _missing = self.beijing_aq[target]['missing'][index_start:index_end]
            missing_index_target.append(torch.from_numpy(_missing.to_numpy()).squeeze(-1).unsqueeze(0))
            x_target.append(torch.from_numpy(_x_target.to_numpy()).unsqueeze(0))

            _y_target = self.beijing_aq[target]['pred'][index_start:index_end]
            y_target.append(torch.from_numpy(_y_target.to_numpy()).unsqueeze(0))
        x_target, y_target = torch.cat(x_target, dim=0), torch.cat(y_target, dim=0)

        for context in self.context_station_list:
            _x_context = self.beijing_aq[context]['feat'][index_start:index_end]
            _missing = self.beijing_aq[context]['missing'][index_start:index_end]
            missing_index_context.append(torch.from_numpy(_missing.to_numpy()).squeeze(-1).unsqueeze(0))
            x_context.append(torch.from_numpy(_x_context.to_numpy()).unsqueeze(0))

            _y_context = self.beijing_aq[context]['pred'][index_start:index_end]
            y_context.append(torch.from_numpy(_y_context.to_numpy()).unsqueeze(0))
        x_context, y_context = torch.cat(x_context, dim=0), torch.cat(y_context, dim=0)

        time = self.beijing_aq[context]['feat'][index_start:index_end].index.values.astype(np.datetime64)
        time = (time - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')

        missing_index_context = torch.cat(missing_index_context, dim=0).permute([1, 0])
        missing_index_target = torch.cat(missing_index_target, dim=0).permute([1, 0])

        if self.opt.phase == 'train':
            t2c_A = self.A[target_station_index, :][:, context_station_index]
        else:
            t2c_A = self.A

        # different models have different settings for adjacency
        if self.opt.use_adj == True:
            norm_A = np.exp(- 0.5 * (t2c_A / np.std(t2c_A, axis=1, keepdims=True)) ** 2)
            adj = torch.from_numpy(norm_A)
            adj_key = 'norm_adj'
        else:
            adj = torch.from_numpy(t2c_A)
            adj_key = 'adj'

        return {'x_context': x_context.float(),  # [num_n, time, d_x]
                'y_context': y_context.float(),  # [num_n, time, d_y]

                'x_target': x_target.float(),  # [num_m, time, d_x]
                'y_target': y_target.float(),  # [num_m, time, d_y]

                adj_key: adj.float(),  # [num_m, num_n]

                'missing_index_context': missing_index_context.float(),  # [time, num_n]
                'missing_index_target': missing_index_target.float(),  # [time, num_m]

                'time': time  # [time]
        }

    def load_loc(self, aq_location_path, context_set, target_set, use_adj=True):
        # load air quality station locations data
        aq_locations_meta = pd.read_csv(aq_location_path)
        aq_locations_meta.head()
        aq_locations_meta.rename(columns={'Unnamed: 0': 'station_id'}, inplace=True)

        # load station locations for adj construction
        beijing_location = aq_locations_meta.drop(columns=['api_data', 'need_prediction', 'historical_data',
                                                           'SiteType', 'SiteName'])
        beijing_location = beijing_location.rename(columns={
            'Latitude': 'latitude',
            'Longitude': 'longitude'
        })
        beijing_location = beijing_location[
            beijing_location['station_id'].isin(self.station_list)].sort_values(
            by=['station_id']).set_index('station_id')

        if use_adj:
            # build adjacency matrix for each target node
            A = np.zeros((len(target_set), len(context_set)))
            for t, target in enumerate(target_set):
                for c, context in enumerate(context_set):
                    dis = self.haversine(beijing_location.at[target, 'longitude'],
                                                beijing_location.at[target, 'latitude'],
                                                beijing_location.at[context, 'longitude'],
                                                beijing_location.at[context, 'latitude'])
                    A[t, c] = dis
        else:
            A = np.zeros((len(target_set), len(context_set), 2))
            for t, target in enumerate(target_set):
                for c, context in enumerate(context_set):
                    A[t, c, 0] = beijing_location.at[target, 'longitude'] - beijing_location.at[context, 'longitude']
                    A[t, c, 1] = beijing_location.at[target, 'latitude'] - beijing_location.at[context, 'latitude']
        return A

    def load_feat(self, data_path, delete_col=None):

        beijing_multimodal = pd.read_csv(data_path, header=0)
        stations_id = beijing_multimodal.station_id.unique().tolist()
        # group by station beijing station id 1001-1036
        beijing_aq = {}
        for id, station_aq in beijing_multimodal.groupby('station_id'):
            station_aq = station_aq.set_index("time").drop(columns=['station_id'])
            if delete_col is not None:
                station_aq = station_aq.drop(columns=delete_col)
            # split data into features and labels
            beijing_aq[id] = {}
            beijing_aq[id]['feat'] = station_aq.drop(columns=self.pred_attrs+self.drop_attrs)
            beijing_aq[id]['missing'] = station_aq[[attr.split('_')[0]+'_Missing' for attr in self.pred_attrs]]
            beijing_aq[id]['pred'] = station_aq[self.pred_attrs]
        return beijing_aq, stations_id
