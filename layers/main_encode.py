from network import *
from data_process import Data_Process
# from models import *
from DLinear import ModelDL
# from PatchTST import Model
# from TiDE import ModelTiDE
# from Transformer import Transformer_vanilla
from transformer_vanilla import RULTransformer
from argparse import Namespace
import torch
import numpy
import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime
import pandas as pd

# Set random seed for reproducibility
seed = 4207
np.random.seed(seed)

class Process():
    def __init__(self, arg : Namespace, model : torch.nn.Module):
        self.arg = arg
        self.data = Data_Process(self.arg)

        self.net = model.to(arg.device)
        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=arg.learning_rate)
        self.model_dir = os.path.join('../results', self.arg.model)
        os.makedirs(self.model_dir, exist_ok=True)
        self.best_rmse = float('inf')



    def Train(self):
        print(f"Starting training of {self.arg.model}...")
        for epoch in range(1, self.arg.epoch+1):

            train_loss = 0
            self.net.train()
            train_dataloader = self.data.getTrainDataloader(
                batch_size=self.arg.batch_size,
                memory_pinned=self.arg.memory_pinned,
            )

            for data, target in train_dataloader:
                data, target = data.to(self.arg.device), target.to(self.arg.device)
                x_mark = data[:, :, -1:]
                output = self.net(data[:,:,0:-1], x_mark, data[:,:,0:-1], x_mark)
                # output = self.net(data, None, None, None)

                self.optimizer.zero_grad()
                loss = self.loss_function(output, target)
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()

                del data, target, output

            aver_loss = train_loss/len(train_dataloader)
            rmse, score = self.Test()
            print('Epoch: {:03d}, '
                  'Train Loss: {:.4f}, Average Train Loss: {:.4f},'
                  'RMSE: {:.4f},'
                  'Score: {:.4f},'.format(epoch, train_loss, aver_loss, rmse.item(), score.item()))
            if rmse < self.best_rmse:
                self.best_rmse = rmse
                print(f"New best model with RMSE: {rmse:.6f}")
                self.save_model(f"{self.arg.model}_best_temp.pth")

        self._log_to_csv(self.arg.model, self.best_rmse, datetime.now())
        path = f"{self.arg.model}_best.pth"
        path_temp = f"{self.arg.model}_best_temp.pth"
        his_best_rmse = self._get_his_best_rmse()
        if self.best_rmse < his_best_rmse:
            self.load_model(path_temp)
            self.save_model(path)
            print(f"Best RMSE upgraded，RMSE: {self.best_rmse}")

    def _get_his_best_rmse(self):
        if not os.path.exists(self.csv_file):
            return float('inf')  # 文件不存在时返回正无穷
        df = pd.read_csv(self.csv_file)
        if df.empty:
            return float('inf')  # 文件为空时返回正无穷
        return df['rmse'].min()

    def Test(self):
        test_dataloader = self.data.getTestDataloader(
            batch_size=self.arg.batch_size,
            memory_pinned=self.arg.memory_pinned,
        )
        self.net.eval()
        predictions = []
        targets = []

        with torch.no_grad():
            for data, target in test_dataloader:
                # Move data to device
                data, target = data.to(self.arg.device), target.to(self.arg.device)
                output = self.net(data[0:-1], data[-1])

                # Store predictions and targets
                predictions.append(output.cpu().numpy())
                targets.append(target.cpu().numpy())
                del data, target, output
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)

        score = self.nasa_score(predictions, targets)

        return rmse, score

    def nasa_score(self, predicted, actual):
        score = 0
        for i in range(len(predicted)):
            if actual[i] > predicted[i]:
                # Late prediction (actual RUL > predicted RUL)
                score += np.exp((actual[i] - predicted[i]) / 13) - 1
            else:
                # Early prediction (actual RUL <= predicted RUL)
                score += np.exp((predicted[i] - actual[i]) / 10) - 1

        return score/len(predicted)

    def save_model(self, filename):
        """
        Save the model to disk

        Args:
            filename: Filename to save the model as
        """
        path = os.path.join(self.model_dir, filename)
        torch.save(self.net.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, filename):
        """
        Load a model from disk

        Args:
            filename: Filename of the model to load
        """
        path = os.path.join(self.model_dir, filename)
        file = torch.load(path)
        self.net.load_state_dict(file)
        self.net.to(self.arg.device)
        print(f"Model loaded from {path}")

    def _log_to_csv(self, model_name, rmse, training_time):
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                [model_name, rmse, training_time, self.arg.dataset, self.arg.input_size, self.arg.window_size,
                 self.arg.sampling, self.arg.skip, self.arg.stride, self.arg.load_data_mode])



def args_config(dataset_choice : int, model_name : str) -> Namespace:
    dataset_list = ['N-CMAPSS_DS01-005', 'N-CMAPSS_DS02-006', 'N-CMAPSS_DS03-012', 'N-CMAPSS_DS04',
                    'N-CMAPSS_DS05', 'N-CMAPSS_DS06', 'N-CMAPSS_DS07', 'N-CMAPSS_DS08a-009', 'N-CMAPSS_DS08c-008']
    unit_index_csv = pandas.read_csv('../unit_index.csv')

    arguments = Namespace(
        directory = './',
        dataset   = dataset_list[dataset_choice],
        choice = dataset_choice,
        epoch     = 100,
        device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        max_rul = 65,
        loadmode='cheat1',
        # input_size = 43,#with all parameters and timestamp
        # input_size = 33,#without simu_settings
        # input_size = 34,#with simu_settings(ds01)
        # input_size = 22, #with 2 virtual sensors, simu_settings(ds01)
        input_size=20,  # with 2 virtual sensors, simu_settings(ds01), without time stamp
        learning_rate = 0.005,
        batch_size = 32,
        model = model_name,
        index_train = numpy.fromstring(
            unit_index_csv[unit_index_csv['File'] == dataset_list[dataset_choice]+'.h5']['Dev Units'].values[0][1:-1],
            dtype=numpy.int8, sep=' '
        ).tolist(),
        index_test  = numpy.fromstring(
            unit_index_csv[unit_index_csv['File'] == dataset_list[dataset_choice]+'.h5']['Test Units'].values[0][1:-1],
            dtype=numpy.int8, sep=' '
        ).tolist(),

        stride = 1,
        sampling = 10,
        window_size = 100,
        skip = 0.2,
        load_data_mode = 'average_in_two',

        memory_pinned = True,
        # REMIND: place model hyperparameters here
    )

    if arguments.loadmode == 'normal':
        arguments.input_size = 20 #offical, with 2virtual sensors
    elif arguments.loadmode == 'cheat1':
        arguments.input_size = 21 # with 2 virtual sensors, simu_settings(ds01), without time stamp
    elif arguments.loadmode == 'encode|cheat2':
        arguments.input_size = 22 # with 2 virtual sensors, simu_settings(ds01) and time stamp

    return arguments

class Configs():
    task_name = 'rul_prediction'
    seq_len = 100
    pred_len = 1
    moving_avg = 25
    enc_in = 20 #default
    dropout = 0.1
    seg_len = 10
    num_class = 10
    d_model = 256
    d_ff = 256
    patience = 5
    e_layers = 2
    d_layers = 2
    dec_in = 7
    c_out = 8
    freq = 'm'
    factor = 3
    output_attention = False
    n_heads = 2
    activation = 'gelu'
    embed = 'timeF'
    channel_independence = 0



def main(dataset_choice, model_name) -> None:
    args = args_config(dataset_choice, model_name)
    configs = Configs()
    configs.enc_in = args.input_size
    if args.model == 'TSMixer':
        model = TSMixer(
            sensors=args.input_size,
            e_layers=12,
            d_model=48,
            seq_len=args.window_size,
            pred_len=1,
            dropout=0.1
        )
    elif args.model == 'LSTM_pTSMixer_GA':
        model = LSTM_pTSMixer_GA(
            sensors=args.input_size, e_layers=8, t_model=36, c_model=36,
            lstm_layer_num=8, seq_len=args.window_size,
            dropout=0.2, accept_window=args.window_size
        )
    elif args.model == 'simple':
        model = SimpleRULPredictor(
            input_size=args.input_size,
            hidden_size=128,
            num_layers=4,
            dropout=0.2
        )
    elif args.model == 'CNN':
        model = CNNRULPredictor(
            input_size=args.input_size,
            sequence_length=args.window_size,
            num_filters=64,
            kernel_size=3,
            dropout=0.2
        )
    elif args.model == 'hybrid':
        model = HybridRULPredictor(
            input_size=args.input_size,
            sequence_length=args.window_size,
            cnn_filters=32,
            lstm_hidden=64,
            dropout=0.2
        )
    elif args.model == 'DLinear':
        model = ModelDL(configs, individual=False)
    elif args.model == 'FreTS':
        model = FreTS(configs)
    elif args.model == 'SegRNN':
        model = SegRNN(configs)
    elif args.model == 'Transformer':
        model = Transformer_v(configs)


    # args.model_name = model.name

    instance = Process(args, model)
    instance.Train()

if __name__ == '__main__':
    main(0, 'SegRNN')
    # df = pd.read_csv('model.csv')
    # for i in range(2):
    #     for j in range(len(df)):
    #         main(0, df.iloc[j]['model'])
    # for i in range(3):
    #     for j in range(len(df)):
    #         main(i, df.iloc[j]['model'])
