from data_set import TrainDataset, TestDataset
from load_data import DataPreparation
from load_data_uncct import DataPreparation_ucct
import numpy
import gc
import os
from glob import glob


from argparse import Namespace
from torch.utils.data import DataLoader

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

class Data_Process():
    def __init__(self, arg : Namespace):
        self.arg = arg
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_filedir = os.path.join(current_dir, 'N-CMAPSS')
        self.loadData(
            arg=self.arg,
            root=self.arg.directory + 'database/' + self.arg.dataset + '/',
            window_size=self.arg.window_size,
            skip=self.arg.skip,
            stride=self.arg.stride,
            sampling=self.arg.sampling,
            index_train=self.arg.index_train,
            index_test=self.arg.index_test,
        )


    def loadData(
            self,
            arg,
            root : str,
            window_size : int,
            skip : float,
            stride : int,
            sampling : int,
            index_train : list,
            index_test : list
    ) -> None:
        '''
        This function create train_sample_np, train_label_np and test_sample_np, test_label_np
        for later dataset construction.
        '''
        dataset_list = glob('../database/*')
        data_path = dataset_list[self.arg.dataset_choice]
        self.load_data_mode = arg.load_data_mode
        if self.arg.num_segments >  0:
            data_preparation = DataPreparation(arg, sampling, sequence_length=window_size, unit_index_train=index_train, unit_index_test=index_test, stride=stride, is_normalize=True, data_filepath=data_path)
            data_dic = data_preparation.prepare_data()
        else:
            data_preparation = DataPreparation_ucct(arg, sampling, sequence_length=window_size, unit_index_train=index_train, unit_index_test=index_test, stride=stride, is_normalize=True, data_filepath=data_path)
            data_dic = data_preparation.prepare_data()
        train_data = data_dic['train']
        test_data = data_dic['test']
        self.train_sample = train_data['samples']
        self.train_label = train_data['labels']
        # self.train_sample, self.train_label = self.skipTime(self.train_sample, self.train_label, self.arg.skip)
        self.test_sample = test_data['samples']
        self.test_label = test_data['labels']
        # self.test_sample, self.test_label = self.skipTime(self.test_sample, self.test_label, self.arg.skip)
        del train_data, test_data
        gc.collect()

    def skipTime(
            self,
            sample_array : numpy.ndarray,
            label_array : numpy.ndarray,
            skip : float
    ) -> (numpy.ndarray, numpy.ndarray):
        """
        If skip=0.1
        This function skips the first 10% and last 10% of the samples for a given RUL of an engine.
        This is done because the features during the change in RUL to the next number are highly noisy for the model to predict
        """
        return_label_array = numpy.array([], dtype=numpy.float32)#.reshape(0, label_array.shape[1])
        return_sample_array = numpy.array([], dtype=numpy.float32).reshape(0, sample_array.shape[1], sample_array.shape[2])
    
        ruls = numpy.unique(label_array.astype(numpy.int8), return_index=True)
        # ruls_ordered = ruls[0][::-1].astype(numpy.float32)
        ruls_ordered_indices = ruls[1][::-1]
    
        split_label_array = numpy.split(label_array, ruls_ordered_indices[1:])
        split_sample_array = numpy.split(sample_array, ruls_ordered_indices[1:])
    
        for i in range(len(split_label_array)):
            sub_array_len = len(split_label_array[i])
            skip_len = int(skip*sub_array_len)
            return_label_array = numpy.concatenate((return_label_array, split_label_array[i][skip_len:-skip_len]))
            return_sample_array = numpy.concatenate((return_sample_array, split_sample_array[i][skip_len:-skip_len]))
        return return_sample_array, return_label_array

    def getTrainDataloader(
            self,
            batch_size : int,
            memory_pinned : bool
    ) -> DataLoader:
        train_dataset = TrainDataset(
            sample=self.train_sample,
            label=self.train_label,
        )

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=2,
            pin_memory=memory_pinned,
            prefetch_factor=2,
            persistent_workers=True,
        )

        return train_dataloader


    def getTestDataloader(
            self,
            batch_size : int,
            memory_pinned : bool
    ) -> DataLoader:
        test_dataset = TestDataset(
            sample=self.test_sample,
            label=self.test_label,
        )

        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=2,
            pin_memory=memory_pinned,
            prefetch_factor=2,
            persistent_workers=True,
        )

        return test_dataloader

