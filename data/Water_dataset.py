from data.base_dataset import BaseDataset
import pandas as pd
import numpy as np
import scipy.io as scio
from sklearn.preprocessing import StandardScaler


class WaterDataset(BaseDataset):
    """
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
        parser.set_defaults(y_dim=1, covariate_dim=2)
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        """
        load data give options
        """
        self.opt = opt

        self.pred_attrs = [opt.pred_attr]
        self.drop_attrs = ['RC_Missing', 'TB_Missing', 'PH_Missing', 'latitude', 'longitude']
        self.cont_cols = ['RC', 'TB', 'PH']

        location_path = 'dataset/water/all_node_cord.mat'
        data_path = 'dataset/water/processed_raw.csv'

        self.A = self.load_loc(location_path)
        self.raw_data, norm_info = self.load_feat(data_path, self.time_division[opt.phase])
        test_nodes_path = 'dataset/water/test_nodes.npy'

        # get data division index
        self.test_node_index = self.get_node_division(test_nodes_path, num_nodes=self.raw_data['pred'].shape[0])
        self.train_node_index = np.setdiff1d(np.arange(self.raw_data['pred'].shape[0]), self.test_node_index)

        # add norm info
        self.add_norm_info(norm_info.at['mean', opt.pred_attr], norm_info.at['scale', opt.pred_attr])

        # data format check
        self._data_format_check()

    # def load_loc(self, loc_path):
    #     print('Loading station locations...')
    #     A = scio.loadmat(loc_path)['wq_sim']
    #     A = np.exp(- 0.5 * (A / np.std(A)) ** 2)
    #     return A

    def load_loc(self, node_cord_path, build_adj=True):
        # load air quality station locations data
        location = pd.DataFrame(scio.loadmat(node_cord_path)['wq_node_cord'])
        location.rename(columns={0: 'latitude', 1: 'longitude'}, inplace=True)
        location = location[['latitude', 'longitude']].astype('float')
        num_station = len(location)

        if build_adj:
            # build adjacency matrix for each target node
            A = np.zeros((num_station, num_station))
            for t in range(num_station):
                for c in range(num_station):
                    dis = self.haversine(location.at[t, 'longitude'],
                                                location.at[t, 'latitude'],
                                                location.at[c, 'longitude'],
                                                location.at[c, 'latitude'])
                    A[t, c] = dis
            # Gaussian and normalization
            A = np.exp(- 0.5 * (A / np.std(A)) ** 2)
        else:
            A = np.zeros((num_station, num_station, 2))
            for t in range(num_station):
                for c in range(num_station):
                    A[t, c, 0] = location.at[t, 'longitude'] - location.at[c, 'longitude']
                    A[t, c, 1] = location.at[t, 'latitude'] - location.at[c, 'latitude']
        return A

    def load_feat(self, data_path, time_division):

        water_data = pd.read_csv(data_path, header=0)
        print('Computing normalization info...')
        feat_scaler = StandardScaler()
        water_data[self.cont_cols] = feat_scaler.fit_transform(water_data[self.cont_cols])
        norm_info = pd.DataFrame([feat_scaler.mean_, feat_scaler.scale_, feat_scaler.var_], columns=self.cont_cols, index=['mean', 'scale', 'var'])

        print('Loading water features...')
        data = {'feat': [],
                'pred': [],
                'missing': [],
                'time': []}
        for id, station_water in water_data.groupby('station_id'):
            station_water = station_water.set_index("time").drop(columns=['station_id'])
            # split data into features and labels
            data['feat'].append(station_water.drop(columns=self.pred_attrs+self.drop_attrs).to_numpy()[np.newaxis])
            data['missing'].append(station_water[[attr.split('_')[0]+'_Missing' for attr in self.pred_attrs]].to_numpy()[np.newaxis])
            data['pred'].append(station_water[self.pred_attrs].to_numpy()[np.newaxis])

        data_length = data['feat'][0].shape[1]
        start_index, end_index = int(time_division[0] * data_length), int(time_division[1] * data_length)
        data['feat'] = np.concatenate(data['feat'], axis=0)[:, start_index:end_index, :]
        data['missing'] = np.concatenate(data['missing'], axis=0)[:, start_index:end_index, :]
        data['pred'] = np.concatenate(data['pred'], axis=0)[:, start_index:end_index, :]
        data['time'] = station_water[start_index:end_index].index.values.astype(np.datetime64)
        data['time'] = ((data['time'] - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's'))
        return data, norm_info
