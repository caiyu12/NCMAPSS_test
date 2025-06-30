import numpy as np
import pandas as pd
import h5py
from sklearn import preprocessing
import numpy


class DataPreparation_ucct:
    def __init__(self, arg, sampling, sequence_length, unit_index_train , unit_index_test, stride, is_normalize, data_filepath):
        self.arg = arg
        self.data_filepath = data_filepath
        self.sampling = sampling
        self.sequence_length = sequence_length
        self.units_index_train = unit_index_train
        self.units_index_test = unit_index_test
        self.is_normalize = is_normalize
        self.stride = stride
        self.df_all = None
        self.df_train = None
        self.df_test = None
        self.sequence_cols = None

    def load_and_prepare_data(self):
        with h5py.File(self.data_filepath, 'r') as hdf:
            W = np.concatenate((hdf.get('W_dev'), hdf.get('W_test')), axis=0)
            X_s = np.concatenate((hdf.get('X_s_dev'), hdf.get('X_s_test')), axis=0)
            X_v = np.concatenate((hdf.get('X_v_dev'), hdf.get('X_v_test')), axis=0)
            Y = np.concatenate((hdf.get('Y_dev'), hdf.get('Y_test')), axis=0)
            A = np.concatenate((hdf.get('A_dev'), hdf.get('A_test')), axis=0)
            T = np.concatenate((hdf.get('T_dev'), hdf.get('T_test')), axis=0) #just for test

            W_var = [str(var, 'utf-8') for var in hdf.get('W_var')]
            X_s_var = [str(var, 'utf-8') for var in hdf.get('X_s_var')]
            X_v_var = [str(var, 'utf-8') for var in hdf.get('X_v_var')]
            A_var = [str(var, 'utf-8') for var in hdf.get('A_var')]
            T_var = [str(var, 'utf-8') for var in hdf.get('T_var')]

        df_W = pd.DataFrame(data=W, columns=W_var)
        df_Xs = pd.DataFrame(data=X_s, columns=X_s_var)
        # df_Xv = pd.DataFrame(data=X_v, columns=X_v_var)
        df_Xv = pd.DataFrame(data=X_v[:, 0:2], columns=['T40', 'P30'])
        df_Y = pd.DataFrame(data=Y, columns=['RUL'])
        df_A = pd.DataFrame(data=A, columns=A_var).drop(columns=['cycle', 'Fc', 'hs'])
        df_T_origin = pd.DataFrame(data=T, columns=T_var)
        df_T = df_T_origin[self.arg.fault_index]
        # df_T = pd.DataFrame(df_T_origin[self.arg.fault_index], columns=[self.arg.fault_index])

        # df_T = pd.DataFrame(data=T[:,-4], columns=['HPT_eff_mod'])

        ts_df_A = self.timestamp_creater(df_A)
        if self.arg.loadmode == 'encode|cheat12':
            self.df_all = pd.concat([ts_df_A, df_T, df_W, df_Xs, df_Xv, df_Y], axis=1)[::self.sampling]
        elif self.arg.loadmode == 'cheat1':
            self.df_all = pd.concat([ts_df_A['unit'],df_T, df_W, df_Xs, df_Xv, df_Y], axis=1)[::self.sampling]
        elif self.arg.loadmode == 'normal':
            self.df_all = pd.concat([ts_df_A['unit'], df_W, df_Xs, df_Xv, df_Y], axis=1)[::self.sampling]
        elif self.arg.loadmode == 'normal1' or self.arg.loadmode == 'encode1':
            self.df_all = pd.concat([ts_df_A, df_W, df_Xs, df_Xv, df_Y], axis=1)[::self.sampling]

    def timestamp_creater(self, df_A):
        df_new = pd.DataFrame()
        df_new['timestamp'] = df_A.groupby('unit').cumcount() + 1
        df_new['unit'] = df_A['unit']

        return df_new

    def split_train_test(self):
        self.df_train = pd.concat([self.df_all[self.df_all['unit'] == unit]
                                  for unit in self.units_index_train]).reset_index(drop=True)
        self.df_test = pd.concat([self.df_all[self.df_all['unit'] == unit]
                                 for unit in self.units_index_test]).reset_index(drop=True)

    def normalize_data(self):
        # 获取原始列中除 'RUL' 和 'unit' 外的列，并保持顺序
        cols_normalize = [col for col in self.df_train.columns if col not in ['RUL', 'unit']]
        # cols_normalize = self.df_train.columns.difference(['RUL', 'unit']).astype(str)
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

        norm_df_train = pd.DataFrame(min_max_scaler.fit_transform(self.df_train[cols_normalize]),
                                    columns=cols_normalize, index=self.df_train.index)
        self.df_train = self.df_train[['RUL', 'unit']].join(norm_df_train)

        norm_df_test = pd.DataFrame(min_max_scaler.transform(self.df_test[cols_normalize]),
                                   columns=cols_normalize, index=self.df_test.index)
        self.df_test = self.df_test[['RUL', 'unit']].join(norm_df_test)
        self.sequence_cols = cols_normalize

    def generate_sequences(self, df):
        sample_list, label_list, unit_list = [], [], []

        for unit in df['unit'].unique():
            unit_df = df[df['unit'] == unit]
            data_matrix = unit_df[self.sequence_cols].values
            label_matrix = unit_df['RUL'].values
            label_matrix = self.cap_label_matrix(label_matrix, max_rul= 65)
            num_elements = data_matrix.shape[0]

            for start in range(0, num_elements - self.sequence_length + 1, self.stride):
                stop = start + self.sequence_length
                sample_list.append(data_matrix[start:stop, :])
                label_list.append(label_matrix[stop - 1])
                unit_list.append(unit)

        return (np.array(sample_list, dtype=np.float32),
                np.array(label_list, dtype=np.float32),
                np.array(unit_list))

    def generate_data_and_labels(self, df):
        """生成并合并所有 unit 的数据"""
        data_list = []
        label_list = []
        unit_list = []

        for unit in df['unit'].unique():
            unit_df = df[df['unit'] == unit]
            data_matrix = unit_df[self.sequence_cols].values
            label_matrix = unit_df['RUL'].values
            label_matrix = self.cap_label_matrix(label_matrix, max_rul=65)
            num_elements = data_matrix.shape[0]

            unique_labels, unique_indices = np.unique(label_matrix, return_index=True)
            unique_labels = unique_labels[::-1]
            unique_indices = unique_indices[::-1]

            for i in range(len(unique_labels) - 1):
                start = unique_indices[i]
                stop = unique_indices[i + 1]
                segment_data = data_matrix[start:stop, :]
                segment_data = segment_data[int(0.2 * len(segment_data)):int((1 - 0.2) * len(segment_data))]
                data_list.append(segment_data)
                label_list.extend([unique_labels[i]] * len(segment_data))
                unit_list.extend([unit] * len(segment_data))
                # label_list.append(unique_labels[i])
                # unit_list.append(unit)

        # 转换为 DataFrame
        data_array = np.vstack(data_list)
        data_df = pd.DataFrame(data_array, columns=self.sequence_cols)
        data_df['RUL'] = label_list
        data_df['unit'] = unit_list

        return data_df

    def generate_sequence(self, data_df):
        """按 unit 分组并进行滑窗操作"""
        sample_list = []
        label_list = []
        unit_list = []

        grouped = data_df.groupby('unit')

        for unit, group in grouped:
            data_matrix = group[self.sequence_cols].values
            label_matrix = group['RUL'].values

            for start in range(0, len(data_matrix) - self.sequence_length + 1, self.stride):
                stop = start + self.sequence_length
                sample = data_matrix[start:stop, :]
                label = label_matrix[stop - 1]
                sample_list.append(sample)
                label_list.append(label)
                unit_list.append(unit)

        return (np.array(sample_list, dtype=np.float32),
                np.array(label_list, dtype=np.float32),
                np.array(unit_list, dtype=np.float32))

    def cap_label_matrix(self, label_matrix, max_rul):
        return np.clip(label_matrix, a_min=None, a_max=max_rul)

    def prepare_data(self):
        self.load_and_prepare_data()
        self.split_train_test()
        if self.is_normalize:
            self.normalize_data()
        # train_samples, train_labels, train_units = self.generate_sequences(self.df_train)
        # test_samples, test_labels, test_units = self.generate_sequences(self.df_test)

        train_data_df = self.generate_data_and_labels(self.df_train)
        test_data_df = self.generate_data_and_labels(self.df_test)
        train_samples, train_labels, train_units = self.generate_sequence(train_data_df)
        test_samples, test_labels, test_units = self.generate_sequence(test_data_df)

        return {
            'train': {
                'samples': train_samples,
                'labels': train_labels,
                'units': train_units
            },
            'test': {
                'samples': test_samples,
                'labels': test_labels,
                'units': test_units
            }
        }