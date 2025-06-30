from network import *
from SegRNNs import *
from data_process import Data_Process
# from models import *
# from DLinear import ModelDL
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
import random
from matplotlib.colors import LinearSegmentedColormap

from network.SegRNN import SegRNN


class Process():
    def __init__(self, arg : Namespace, model : torch.nn.Module):
        self.arg = arg
        self.data = Data_Process(self.arg)

        self.net = model.to(arg.device)
        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=arg.learning_rate)
        self.model_dir = os.path.join('./results', self.arg.model)

        self.visualization_dir = os.path.join('.', 'visualization')
        os.makedirs(self.visualization_dir, exist_ok=True)

        os.makedirs(self.model_dir, exist_ok=True)
        self.best_rmse = float('inf')
        self.csv_file = os.path.join('./results', "TC_pRNN_on_no_aver.csv")
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["model_name", "rmse", "training_time", "dataset", "channels", "window_size", "sample", "skip", "stride","load_data_mode", "loadmode","seed"])

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
                output = self.net(data)

                self.optimizer.zero_grad()
                loss = self.loss_function(output, target)
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()

                del data, target, output

            aver_loss = train_loss/len(train_dataloader)
            rmse, score, predictions, targets = self.Test()
            # print('Epoch: {:03d}, '
            #       'Train Loss: {:.4f}, Average Train Loss: {:.4f},'
            #       'RMSE: {:.4f},'
            #       'Score: {:.4f},'.format(epoch, train_loss, aver_loss, rmse.item(), score.item()))
            print('Epoch: {:03d}, '
                  'Train Loss: {:.4f},'
                  'RMSE: {:.4f},'
                  'Score: {:.4f},\n'.format(epoch, train_loss, rmse.item(), score.item())
                  )
            if rmse < self.best_rmse:
                self.best_rmse = rmse
                print(f"New best model with RMSE: {rmse:.6f}")
                self.save_model(f"{self.arg.model}_best_temp_in_ds0{self.arg.choice + 1}.pth")

        path = f"{self.arg.model}_best_in_ds0{self.arg.choice + 1}.pth"
        path_temp = f"{self.arg.model}_best_temp_in_ds0{self.arg.choice + 1}.pth"
        his_best_rmse = self._get_his_best_rmse(self.arg.model)
        print(f"Historical Best RMSE: {his_best_rmse}")
        if self.best_rmse < his_best_rmse:
            self.load_model(path_temp)
            self.save_model(path)
            best_rmse, best_score, best_predictions, best_targets = self.Test()
            self.visualization(best_predictions, best_targets, best_rmse,  best_score)
            print(f"Best RMSE upgraded，RMSE: {self.best_rmse}")
        self._log_to_csv(self.arg.model, self.best_rmse, datetime.now())

    def _get_his_best_rmse(self, model_name):
        if not os.path.exists(self.csv_file):
            return float('inf')  # 文件不存在时返回正无穷
        df = pd.read_csv(self.csv_file)
        filtered_model_df = df[df['model_name'] == model_name]
        filtered_set_df = filtered_model_df[filtered_model_df['dataset'] == self.arg.dataset]
        if filtered_set_df.empty:
            return float('inf')
        if df.empty:
            return float('inf')  # 文件为空时返回正无穷
        min_rmse = filtered_set_df['rmse'].min()
        return min_rmse

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
                output = self.net(data)

                # Store predictions and targets
                predictions.append(output.cpu().numpy())
                targets.append(target.cpu().numpy())
                del data, target, output
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)

        score = self.nasa_score(predictions, targets)

        return rmse, score, predictions, targets

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

    def visualization(self, prediction, real, rmse, score):
        dataset_name = self.arg.dataset
        model_name = self.arg.model
        save_dir = os.path.join(self.visualization_dir, dataset_name, model_name)
        os.makedirs(save_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(real, color='blue', label='Real RUL', linewidth=2)
        ax.plot(prediction, color='red', label='Predicted RUL', linestyle='--', linewidth=2)
        ax.set_title(f'RUL Prediction for {dataset_name} using {model_name}\nRMSE: {rmse.item():.4f}, Score: {score.item():.4f}',
                     fontsize=14)
        ax.set_xlabel('Time Steps', fontsize=12)
        ax.set_ylabel('RUL', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{dataset_name}_{model_name}_RMSE{rmse:.4f}_{timestamp}.png'
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Visualization saved to {save_path}")

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
            writer.writerow([model_name, rmse, training_time, self.arg.dataset, self.arg.input_size,
                             self.arg.window_size, self.arg.sampling, self.arg.skip, self.arg.stride, self.arg.load_data_mode, self.arg.loadmode, self.arg.random_seed])



def args_config(dataset_choice : int, model_name : str) -> Namespace:
    dataset_list = ['N-CMAPSS_DS01-005', 'N-CMAPSS_DS02-006', 'N-CMAPSS_DS03-012', 'N-CMAPSS_DS04',
                    'N-CMAPSS_DS05', 'N-CMAPSS_DS06', 'N-CMAPSS_DS07', 'N-CMAPSS_DS08a-009', 'N-CMAPSS_DS08c-008']
    unit_index_csv = pandas.read_csv('unit_index.csv')
    df_faults = pd.read_csv('fault_mode.csv')
    input_size_add = df_faults['dataset'].value_counts().sort_index()
    fault_mode_index = df_faults.groupby('dataset')['fault_mode'].apply(list).to_dict()
    fault_index_list = fault_mode_index[dataset_choice+1]

    arguments = Namespace(
        directory = './',
        dataset   = dataset_list[dataset_choice],
        choice = dataset_choice,
        epoch     = 150,
        device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        max_rul = 65,
        # input_size = 43,#with all parameters and timestamp
        # input_size = 33,#without simu_settings
        # input_size = 34,#with simu_settings(ds01)
        # input_size = 22, #with 2 virtual sensors, simu_settings(ds01)
        input_size=20,  # with 2 virtual sensors, simu_settings(ds01), without time stamp
        learning_rate = 0.001,
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
        window_size = 50,
        skip = 0.2,
        load_data_mode = 'no_average', #average_in_two,average_in_one,no_average
        loadmode='normal',
        fault_index = fault_index_list,
        random_seed = 42,

        memory_pinned = True,
        # REMIND: place model hyperparameters here
    )

    if arguments.loadmode == 'normal':
        arguments.input_size = 20 #offical, with 2 virtual sensors
    elif arguments.loadmode == 'cheat1':
        arguments.input_size = 20 + input_size_add[dataset_choice+1] # with 2 virtual sensors, simu_settings(ds01), without time stamp
    elif arguments.loadmode == 'encode|cheat12':
        arguments.input_size = 21 + input_size_add[dataset_choice+1] # with 2 virtual sensors, simu_settings(ds01) and time stamp

    return arguments

class Configs():
    task_name = 'rul_prediction'
    seq_len = 100
    pred_len = 1
    moving_avg = 25
    enc_in = 20 #default
    cov_dim = 4
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
    mixer_kernel_size = 8
    patch_len = 10
    stride = 2
    hidden_dim_cov = 16
    decoder_layers = 3



def main(dataset_choice, model_name) -> None:
    random_seed = random.randint(0, 10000)
    # random_seed = 1260
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    args = args_config(dataset_choice, model_name)
    args.random_seed = random_seed
    if args.load_data_mode == 'no_average':
        args.batch_size = 1024
        args.epoch = 50
        args.sampling = 100
    configs = Configs()
    configs.enc_in = args.input_size
    configs.seq_len = args.window_size
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
            sensors=int(args.input_size), e_layers=8, t_model=36, c_model=36,
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
        model = ModelDL(configs, individual=True)
    elif args.model == 'FreTS':
        model = FreTS(configs)
    elif args.model == 'SegRNN':
        ori_d = configs.d_model
        configs.d_model = 256
        model = SegRNN(configs)
        configs.d_model = ori_d
    elif args.model == 'Transformer':
        model = Transformer_v(configs)
    elif args.model == 'PatchMixer':
        ori_d = configs.d_model
        configs.d_model = 64
        model = ModelPM(configs)
        configs.d_model = ori_d
    elif args.model == 'FC_STGNN':
        model = FC_STGNN_RUL(
            patch_size=2,  # 补丁大小
            num_patch=args.window_size/2,  # 补丁数量
            encoder_time_out=4,  # 编码器输出的时间步数
            encoder_hidden_dim=8,  # 编码器隐藏层维度
            encoder_out_dim=32,  # 编码器输出维度
            encoder_conv_kernel=2,  # 卷积核大小
            hidden_dim=8,  # 图神经网络隐藏层维度
            num_sequential=6,  # 图神经网络处理的时间步数
            num_node=args.input_size,  # 节点数（传感器数量）
            num_windows=36  # 图神经网络窗口数量
        )
    elif args.model == 'SegRNN_TSMixer':
        ori_d = configs.d_model
        configs.d_model = 256
        model = SegRNN_pTSMixer(configs)
        configs.d_model = ori_d
    elif args.model == 'PatchRNN':
        ori_d = configs.d_model
        configs.d_model = 256
        model = PatchRNN(configs)
        configs.d_model = ori_d
    elif args.model == 'pRNN':
        ori_d = configs.d_model
        configs.d_model = 256
        model = pRNN(configs)
        configs.d_model = ori_d
    elif args.model == 'method1':
        ori_d = configs.d_model
        configs.d_model = 256
        model = Channel_wised_RNN_Decoder(configs)
        configs.d_model = ori_d
    elif args.model == 'method2':
        ori_d = configs.d_model
        configs.d_model = 256
        model = SegRNN_1(configs)
        configs.d_model = ori_d
    elif args.model == 'method3':
        ori_d = configs.d_model
        configs.d_model = 256
        model = CNN_SegRNN(configs)
        configs.d_model = ori_d
    elif args.model == 'SA_SegRNN':
        model = SA_SegRNN(configs)
    elif args.model == 'TC_SA_SegRNN':
        model = TC_SA_SegRNN(configs)
    elif args.model == 'method4':
        model = CNN_SegRNN(configs)
    elif args.model == 'CA_cov_SegRNN':
        model = CA_cov_SegRNN(configs)
    elif args.model == 'FilM_SegRNN':
        model = FilM_SegRNN(configs)
    elif args.model == 'SA_CA_SegRNN':
        model = SA_CA_SegRNN(configs)
    elif args.model == 'SA_pRNN':
        model = SA_pRNN(configs)
    elif args.model == 'CA_pRNN':
        model = CA_pRNN(configs)
    elif args.model == 'TC_pRNN':
        model = TC_pRNN(configs)
    # args.model_name = model.name

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    total_params = sum(p.numel() for p in model_parameters)
    print(f"Total Trainable Parameters: {total_params}")
    instance = Process(args, model)
    instance.Train()

if __name__ == '__main__':
    df = pd.read_csv('model.csv')
    # main(0, 'pRNN')
    # main(0, 'SegRNN')

    for i in range(8):
        main(i, 'Transformer')
    #     main(i, 'SegRNN')
    #     main(i, 'CA_cov_SegRNN')

    # for i in range(4):
    #     main(3, 'SegRNN_TSMixer')
    #     main(3, 'SA_pRNN')
    #     main(3, 'CA_pRNN')
    #     main(3,'TC_pRNN')
    #     main(3,'pRNN')
    #     main(3, 'SegRNN')

    # for i in range(2):
    #     for j in range(len(df)):
    #         main(0, df.iloc[j]['model'])
    # for k in range(5):
    #     for i in range(8):
    #         for j in range(len(df)):
    #             main(i, df.iloc[j]['model'])
    # for i in range(8):
    #     for j in range(len(df)):
    #         main(i, df.iloc[j]['model'])
    # for j in range(len(df)):
    #     main(2, df.iloc[j]['model'])
        # main(3, df.iloc[j]['model'])
    # model.csv
    # TSMixer
    # LSTM_pTSMixer_GA
    # simple
    # CNN
    # hybrid
    # DLinear
    # Transformer
