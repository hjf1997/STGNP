from data.base_dataset import BaseDataset
import torch
import pandas as pd
import numpy as np
import random
import pickle
from sklearn.preprocessing import StandardScaler

class BJAirDataset(BaseDataset):
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
        parser.set_defaults(y_dim=1, covariate_dim=30, spatial_dim=16)
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

        aq_location_path = 'dataset/bjair/NP/stations.csv'
        data_path = 'dataset/bjair/NP/processed_raw.csv'
        meta_path = 'dataset/bjair/NP/meta_data.pkl'
        test_nodes_path = 'dataset/bjair/NP/test_nodes.npy'

        self.A = self.load_loc(aq_location_path, build_adj=opt.use_adj)
        self.raw_data, norm_info = self.load_feature(data_path, meta_path, self.time_division[opt.phase], opt.delete_col)

        # get data division index
        self.test_node_index = self.get_node_division(test_nodes_path, num_nodes=self.raw_data['pred'].shape[0])
        self.train_node_index = np.setdiff1d(np.arange(self.raw_data['pred'].shape[0]), self.test_node_index)

        # add norm info
        self.add_norm_info(norm_info.at['mean', opt.pred_attr], norm_info.at['scale', opt.pred_attr])

        # data format check
        self._data_format_check()

    def load_loc(self, aq_location_path, build_adj=True):
        """
        Args:
            build_adj: if True, build adjacency matrix else return horizontal and vertical distance matrix
        Returns:

        """
        print('Loading station locations...')
        # load air quality station locations data
        beijing_location = pd.read_csv(aq_location_path)

        # load station locations for adj construction
        beijing_location = beijing_location.sort_values(by=['station_id'])
        num_station = len(beijing_location)

        if build_adj:
            # build adjacency matrix for each target node
            A = np.zeros((num_station, num_station))
            for t in range(num_station):
                for c in range(num_station):
                    dis = self.haversine(beijing_location.at[t, 'longitude'],
                                                beijing_location.at[t, 'latitude'],
                                                beijing_location.at[c, 'longitude'],
                                                beijing_location.at[c, 'latitude'])
                    A[t, c] = dis
            # Gaussian and normalization
            A = np.exp(- 0.5 * (A / np.std(A)) ** 2)
        else:
            A = np.zeros((num_station, num_station, 2))
            for t in range(num_station):
                for c in range(num_station):
                    A[t, c, 0] = beijing_location.at[t, 'longitude'] - beijing_location.at[c, 'longitude']
                    A[t, c, 1] = beijing_location.at[t, 'latitude'] - beijing_location.at[c, 'latitude']
        return A

    def load_feature(self, data_path, meta_path, time_division, delete_col=None):
        beijing_multimodal = pd.read_csv(data_path, header=0)
        # get normalization info
        # sorry we also involve val, test data in normalization, due to the coding complexity
        print('Computing normalization info...')
        with open(meta_path, 'rb') as f:
            cont_cols = pickle.load(f)['cont_cols']  # list
        feat_scaler = StandardScaler()
        beijing_multimodal[cont_cols] = feat_scaler.fit_transform(beijing_multimodal[cont_cols])
        norm_info = pd.DataFrame([feat_scaler.mean_, feat_scaler.scale_, feat_scaler.var_], columns=cont_cols, index=['mean', 'scale', 'var'])

        print('Loading air quality features...')
        # group by station beijing station id 1001-1036
        data = {'feat': [],
                'pred': [],
                'missing': [],
                'time': []}
        for id, station_aq in beijing_multimodal.groupby('station_id'):
            station_aq = station_aq.set_index("time").drop(columns=['station_id'])
            if delete_col is not None:
                station_aq = station_aq.drop(columns=delete_col)
            # split data into features and labels
            data['feat'].append(station_aq.drop(columns=self.pred_attrs+self.drop_attrs).to_numpy()[np.newaxis])
            data['missing'].append(station_aq[[attr.split('_')[0]+'_Missing' for attr in self.pred_attrs]].to_numpy()[np.newaxis])
            data['pred'].append(station_aq[self.pred_attrs].to_numpy()[np.newaxis])
            # data['time'].append()

        data_length = data['feat'][0].shape[1]
        start_index, end_index = int(time_division[0] * data_length), int(time_division[1] * data_length)
        data['feat'] = np.concatenate(data['feat'], axis=0)[:, start_index:end_index, :]
        data['missing'] = np.concatenate(data['missing'], axis=0)[:, start_index:end_index, :]
        data['pred'] = np.concatenate(data['pred'], axis=0)[:, start_index:end_index, :]
        data['time'] = station_aq[start_index:end_index].index.values.astype(np.datetime64)
        data['time'] = ((data['time'] - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's'))
        return data, norm_info
