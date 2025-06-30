import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime
import random
from argparse import ArgumentParser, Namespace
import gc

from network import * # Placeholder for actual imports
from SegRNNs import * # Placeholder for actual imports
from models import *
from data_process import Data_Process

# --- 常量定义 ---
DATASET_LIST = ['N-CMAPSS_DS01-005', 'N-CMAPSS_DS02-006', 'N-CMAPSS_DS03-012', 'N-CMAPSS_DS04',
                'N-CMAPSS_DS05', 'N-CMAPSS_DS06', 'N-CMAPSS_DS07', 'N-CMAPSS_DS08a-009', 'N-CMAPSS_DS08c-008']
DEFAULT_RESULTS_DIR = './results'
DEFAULT_VISUALIZATION_DIR = './visualization'
DEFAULT_UNIT_INDEX_CSV = 'unit_index.csv'
DEFAULT_FAULT_MODE_CSV = 'fault_mode.csv'
DEFAULT_LOG_FILENAME = 'all_experiments_log.csv' # Default name for the unified log file

# --- 定义要记录到CSV的核心列 ---
# 包含用户指定的关键参数和其他重要信息
CSV_LOG_COLUMNS = [
    'timestamp', 'model', 'dataset', 'best_rmse', 'final_score', 'random_seed',
    'epoch', 'batch_size', 'learning_rate',
    'input_size', 'window_size', 'sampling', 'skip', 'load_data_mode', 'loadmode',
    'd_model']


class Process():
    def __init__(self, args: Namespace, model: torch.nn.Module, total_params: int): # Add total_params
        self.args = args
        self.total_params = total_params # Store total params
        self.data = Data_Process(self.args) # Data_Process 需要接收 Namespace 对象

        self.net = model.to(args.device)
        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.learning_rate)

        # 模型特定的保存目录仍然保留
        self.model_dir = os.path.join(args.results_dir, args.dataset, args.model) # Group by dataset then model
        self.visualization_dir = os.path.join(args.visualization_dir, args.dataset, args.model) # Group by dataset then model
        # 使用传入的统一日志文件路径
        self.csv_file = args.log_file

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.visualization_dir, exist_ok=True)
        # 确保日志文件所在的目录也存在
        os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)

        self.best_rmse = float('inf')
        self._ensure_csv_header()

    def _ensure_csv_header(self):
        # 检查统一的 CSV 文件是否存在，如果不存在则写入预定义的表头
        try:
            if not os.path.exists(self.csv_file) or os.path.getsize(self.csv_file) == 0:
                 with open(self.csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(CSV_LOG_COLUMNS) # 使用预定义的列名
        except IOError as e:
            print(f"Error checking or writing CSV header {self.csv_file}: {e}")


    def Train(self):
        print(f"Starting training of {self.args.model} on {self.args.dataset}...")
        start_time = datetime.now()

        for epoch in range(1, self.args.epoch + 1):
            train_loss = 0
            self.net.train()
            train_dataloader = self.data.getTrainDataloader(
                batch_size=self.args.batch_size,
                memory_pinned=self.args.memory_pinned,
            )

            for data, target in train_dataloader:
                data, target = data.to(self.args.device), target.to(self.args.device)
                output = self.net(data)
                self.optimizer.zero_grad()
                loss = self.loss_function(output, target.view_as(output))
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                del data, target, output
            # gate_value = self.net.gate_value.item()
            # cluster_map = self.net.cluster_map
            rmse, score, _, _ = self.Test()
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss / len(train_dataloader):.4f}, Val RMSE: {rmse:.4f}, Val Score: {score:.4f}')
            # print(f"Model gate value is: {gate_value}")
            # print(cluster_map)
            if rmse < self.best_rmse:
                self.best_rmse = rmse
                print(f"  New best model found! RMSE: {self.best_rmse:.6f}. Saving temporarily...")
                self.save_model(f"{self.args.model}_temp_seed{self.args.random_seed}.pth") # Include seed in temp name

            del train_dataloader
            gc.collect()
            if 'cuda' in self.args.device:
                torch.cuda.empty_cache()


        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        print(f"Training finished in {training_time:.2f} seconds.")

        # --- Final Evaluation and Logging ---
        final_model_path = f"{self.args.model}_final_seed{self.args.random_seed}.pth" # Include seed in final name
        temp_model_path_rel = f"{self.args.model}_temp_seed{self.args.random_seed}.pth"
        temp_model_path_abs = os.path.join(self.model_dir, temp_model_path_rel)

        final_rmse = float('inf')
        final_score = float('inf')

        if os.path.exists(temp_model_path_abs):
             print(f"Loading best model from {temp_model_path_abs} for final evaluation...")
             self.load_model(temp_model_path_rel) # Load relative path
             final_rmse, final_score, best_predictions, best_targets = self.Test()
             print(f"Final Best Model -> RMSE: {final_rmse:.6f}, Score: {final_score:.4f}")
             self.save_model(final_model_path)
             self.visualization(best_predictions, best_targets, final_rmse, final_score)
             try:
                 os.remove(temp_model_path_abs)
                 # print(f"Removed temporary model: {temp_model_path_abs}")
             except OSError as e:
                 print(f"Error removing temporary model {temp_model_path_abs}: {e}")
        else:
             print("No temporary model was saved during training (possibly no improvement found).")
             # Still log results even if no improvement, using last epoch's values or inf
             final_rmse = self.best_rmse # Use the best recorded RMSE even if not from saved model
             # final_score = ? # Need to decide how to get score if no model loaded

        # Log results regardless of whether a temp model was saved
        self._log_to_csv(final_rmse, final_score, training_time)


    def Test(self):
        # Test method remains largely the same...
        test_dataloader = self.data.getTestDataloader(
            batch_size=self.args.batch_size,
            memory_pinned=self.args.memory_pinned,
        )
        self.net.eval()
        predictions = []
        targets = []

        with torch.no_grad():
            for data, target in test_dataloader:
                data, target = data.to(self.args.device), target.to(self.args.device)
                output = self.net(data)
                predictions.append(output.cpu().numpy())
                targets.append(target.cpu().numpy())
                del data, target, output

        predictions = np.concatenate(predictions).flatten()
        targets = np.concatenate(targets).flatten()

        if len(predictions) == 0 or len(targets) == 0:
             print("Warning: No predictions or targets generated during testing.")
             return float('inf'), float('inf'), np.array([]), np.array([])

        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse) if mse != float('inf') else float('inf')
        score = self.nasa_score(predictions, targets)

        del test_dataloader
        gc.collect()
        if 'cuda' in self.args.device:
            torch.cuda.empty_cache()

        return rmse, score, predictions, targets

    def nasa_score(self, predicted, actual):
        # NASA score function remains the same...
        score = 0
        if len(predicted) != len(actual):
            print(f"Warning: Predicted length ({len(predicted)}) and actual length ({len(actual)}) mismatch in nasa_score.")
            return float('inf')

        for i in range(len(predicted)):
             actual_val = actual[i].item() if isinstance(actual[i], torch.Tensor) else actual[i]
             predicted_val = predicted[i].item() if isinstance(predicted[i], torch.Tensor) else predicted[i]
             diff = actual_val - predicted_val
             if diff > 0:
                 score += np.exp(diff / 13.0) - 1
             else:
                 score += np.exp(-diff / 10.0) - 1
        return score / len(predicted) if len(predicted) > 0 else 0


    def visualization(self, prediction, real, rmse, score):
        # Visualization function remains largely the same...
        dataset_name = self.args.dataset
        model_name = self.args.model
        save_dir = self.visualization_dir

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(real.flatten(), color='royalblue', label='Real RUL', linewidth=1.5, alpha=0.8)
        ax.plot(prediction.flatten(), color='crimson', label='Predicted RUL', linestyle='--', linewidth=1.5, alpha=0.8)
        title = f'RUL Prediction: {dataset_name} / {model_name} (Seed: {self.args.random_seed})\nRMSE: {rmse:.4f}, Score: {score:.4f}'
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Time Steps', fontsize=14)
        ax.set_ylabel('Remaining Useful Life (RUL)', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=12)
        plt.tight_layout()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{dataset_name}_{model_name}_Seed{self.args.random_seed}_RMSE{rmse:.4f}_{timestamp}.png' # Add seed
        save_path = os.path.join(save_dir, filename)
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        except Exception as e:
            print(f"Error saving visualization: {e}")
        plt.close(fig)


    def save_model(self, filename):
        # save_model function remains the same...
        path = os.path.join(self.model_dir, filename)
        try:
            torch.save(self.net.state_dict(), path)
        except Exception as e:
            print(f"Error saving model to {path}: {e}")


    def load_model(self, filename):
        # load_model function remains the same...
        path = os.path.join(self.model_dir, filename)
        if os.path.exists(path):
            try:
                self.net.load_state_dict(torch.load(path, map_location=self.args.device))
                self.net.to(self.args.device)
                print(f"Model loaded from {path}")
            except Exception as e:
                print(f"Error loading model from {path}: {e}")
        else:
            print(f"Model file not found: {path}")


    def _log_to_csv(self, best_rmse, final_score, training_time):
        # 修改日志记录，只写入预定义的列
         try:
             # Check if file exists to determine if header needs writing (handled in _ensure_csv_header)
             with open(self.csv_file, 'a', newline='') as f:
                 writer = csv.writer(f)
                 # Prepare data row based on predefined CSV_LOG_COLUMNS
                 log_data_row = []
                 current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                 for col in CSV_LOG_COLUMNS:
                     if col == 'timestamp':
                         log_data_row.append(current_timestamp)
                     elif col == 'best_rmse':
                         log_data_row.append(f"{best_rmse:.6f}" if best_rmse != float('inf') else 'inf')
                     elif col == 'final_score':
                         log_data_row.append(f"{final_score:.4f}" if final_score != float('inf') else 'inf')
                     else:
                         # Get value from args, provide default 'N/A' if attribute missing
                         log_data_row.append(getattr(self.args, col, 'N/A'))

                 writer.writerow(log_data_row)
                 # print(f"Results logged to {self.csv_file}") # Optional: confirm logging

         except IOError as e:
             print(f"Error logging results to CSV {self.csv_file}: {e}")
         except Exception as e:
             print(f"An unexpected error occurred during CSV logging: {e}")


# --- 主函数 ---
def main(args: Namespace) -> None:
    # 设置随机种子
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # --- 模型选择与实例化 (与上一版本类似) ---
    model_factory = {
        'simple': lambda args: SimpleRULPredictor(input_size=args.input_size, hidden_size=args.d_model, num_layers=args.d_layers, dropout=args.dropout),
        'CNN': lambda args: CNNRULPredictor(input_size=args.input_size, sequence_length=args.window_size, num_filters=args.d_model, kernel_size=3, dropout=args.dropout),
        'hybrid': lambda args: HybridRULPredictor(input_size=args.input_size, sequence_length=args.window_size, cnn_filters=args.d_ff // 2, lstm_hidden=args.d_model, dropout=args.dropout),
        'TSMixer': lambda args: TSMixer(sensors=args.input_size, e_layers=args.e_layers, d_model=args.d_model, seq_len=args.window_size, pred_len=args.pred_len, dropout=args.dropout),
        'LSTM_pTSMixer_GA': lambda args: LSTM_pTSMixer_GA(sensors=int(args.input_size), e_layers=args.e_layers, t_model=args.d_model//2, c_model=args.d_model//2, lstm_layer_num=args.d_layers, seq_len=args.window_size, dropout=args.dropout, accept_window=args.window_size),
        'DLinear': lambda args: ModelDL(args),
        'FreTS': lambda args: FreTS(args),
        'SegRNN': lambda args: SegRNN(args),
        'Transformer': lambda args: Transformer_v(args),
        'PatchMixer': lambda args: ModelPM(args),
        'SegRNN_TSMixer': lambda args: SegRNN_pTSMixer(args),
        'pRNN': lambda args: pRNN(args),
        'SA_SegRNN': lambda args: SA_SegRNN(args),
        'TC_SA_SegRNN': lambda args: TC_SA_SegRNN(args),
        'CA_cov_SegRNN': lambda args: CA_cov_SegRNN(args),
        'SA_CA_SegRNN': lambda args: SA_CA_SegRNN(args),
        'SA_pRNN': lambda args: SA_pRNN(args),
        'CA_pRNN': lambda args: CA_pRNN(args),
        'TC_pRNN': lambda args: TC_pRNN(args),
        'CNN_pRNN': lambda args: CNN_pRNN(args),
        'CSC_pRNN': lambda args: CSC_pRNN(args),
        'PatchTST': lambda args: ModelPTST(args),
        'SA_CA_pRNN': lambda args: SA_CA_pRNN(args),
        'poolRNN': lambda args: poolRNN(args),
        'pRNN_sq':  lambda args: pRNN_sq(args),
        'pool_pRNN': lambda args: pool_pRNN(args),
        'p2pRNN':  lambda args: p2pRNN(args),
        'poolRNN2': lambda args: poolRNN2(args),
        'poolRNN3': lambda args: poolRNN3(args),
        'p_CNN_RNN': lambda args: p_CNN_RNN(args),
        'p_CNN_MLP_RNN': lambda args: p_CNN_MLP_RNN(args),
        'PatchTSMixer': lambda args: SimplePatchTSMixer(args),
        'SegRNN_CNN': lambda args: SegRNN_CNN(args),
        'SegRNN_GA':  lambda args: SegRNN_GA(args),
        'Simple_SegRNN': lambda args: Simple_SegRNN(args),
        'SegRNN_variant': lambda args: SegRNN_variant(args),
        'pRNN_variant': lambda args: pRNN_variant(args),
        'ParallelRNN': lambda args: ParallelRNN(args),
        'p2p_RNN': lambda args: p2p_RNN(args),
        'CA_ParallelRNN':  lambda args: CA_ParallelRNN(args),
        'CID_RNN':  lambda args: CID_RNN(args),
        'Cluster_RNN':  lambda args: Cluster_RNN(args),
        'IDC_RNN':  lambda args: IDC_RNN(args),
        'iCluster_RNN' : lambda args: Cluster_RNN_improved(args),
        'DUET_RNN': lambda args: AdaptiveDuetRNN(args),
        'PRISM_RNN': lambda args: PRISM_RNN(args),
    }

    if args.model in model_factory:
        model = model_factory[args.model](args)
    else:
        raise ValueError(f"Unknown model: {args.model}. Available models: {list(model_factory.keys())}")

    # 计算可训练参数
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    total_params = sum(p.numel() for p in model_parameters)

    print(f"--- Experiment Setup ---")
    print(f"Log File: {args.log_file}") # Show log file being used
    print(f"Model: {args.model}, Dataset: {args.dataset} (Choice: {args.dataset_choice}), Seed: {args.random_seed}")
    print(f"Total Trainable Parameters: {total_params:,}")
    print(f"Using device: {args.device}")
    print(f"Epochs: {args.epoch}, Batch Size: {args.batch_size}, LR: {args.learning_rate}")
    print(f"Window: {args.window_size}, Input: {args.input_size}, Sampling: {args.sampling}, Skip: {args.skip}, Stride: {args.stride}")
    print(f"Load Mode: {args.loadmode}, Data Mode: {args.load_data_mode}")
    print(f"-------------------------")

    # --- 训练流程 ---
    # Pass total_params to Process instance
    instance = Process(args, model, total_params)
    instance.Train()

# --- 参数解析与执行 ---
if __name__ == '__main__':
    parser = ArgumentParser(description="Train and evaluate RUL prediction models using N-CMAPSS dataset.")

    # --- 基本设置 ---
    random_seed = random.randint(0, 10000)
    parser.add_argument('--model', type=str, default='simple', help='Name of the model to train/evaluate.')
    parser.add_argument('--dataset_choice', type=int, default=0, choices=range(len(DATASET_LIST)), help=f'Index of the N-CMAPSS dataset to use (0-{len(DATASET_LIST)-1}).')
    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Computation device (e.g., "cuda:0", "cpu").')
    parser.add_argument('--random_seed', type=int, default=random_seed, help='Random seed for reproducibility.')
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_DIR, help='Base directory for saving models and visualizations.')
    parser.add_argument('--visualization_dir', type=str, default=DEFAULT_VISUALIZATION_DIR, help='Base directory for saving visualization plots.')
    # 新增：统一日志文件参数
    parser.add_argument('--log_file', type=str, default=os.path.join(DEFAULT_RESULTS_DIR, DEFAULT_LOG_FILENAME), help='Path to the unified CSV file for logging all experiment results.')


    # --- 训练超参数 ---
    parser.add_argument('--epoch', type=int, default=150, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and testing.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Optimizer learning rate.')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping (if implemented).')

    # --- 数据加载与预处理参数 (这些是用户关心的关键参数) ---
    parser.add_argument('--window_size', type=int, default=100, help='Size of the input sequence window (seq_len).')
    parser.add_argument('--input_size', type=int, default=20, help='Number of input features (enc_in, auto-adjusted based on loadmode).')
    parser.add_argument('--stride', type=int, default=1, help='Stride for creating sequences in data loading.')
    parser.add_argument('--sampling', type=int, default=10, help='Sampling rate for reading raw data points.')
    parser.add_argument('--skip', type=float, default=0.2, help='Fraction of data to skip at the beginning/end of each RUL segment (0 to 0.5).')
    parser.add_argument('--load_data_mode', type=str, default='average_in_two', help='Data loading averaging mode.')
    parser.add_argument('--loadmode', type=str, default='cheat1', choices=['normal', 'cheat1', 'encode|cheat12', 'normal1', 'encode1'], help='Feature set loading mode.')
    parser.add_argument('--max_rul', type=int, default=65, help='Maximum RUL value for clipping.')
    parser.add_argument('--memory_pinned', action='store_true', help='Use pinned memory for DataLoader (useful for GPU).')
    parser.add_argument('--unit_index_csv', type=str, default=DEFAULT_UNIT_INDEX_CSV, help='Path to unit index CSV file.')
    parser.add_argument('--fault_mode_csv', type=str, default=DEFAULT_FAULT_MODE_CSV, help='Path to fault mode CSV file.')
    parser.add_argument('--num_segments', type=int, default=2, help='Seg and do the averagepooling.')
    parser.add_argument('--directory', type=str, default='./', help='Directory for loading data.')


    # --- 模型特定超参数 (根据需要调整默认值或在脚本中覆盖) ---
    parser.add_argument('--pred_len', type=int, default=1, help='Prediction length (usually 1 for RUL).')
    parser.add_argument('--d_model', type=int, default=256, help='Dimension of the model embedding/hidden states.')
    parser.add_argument('--d_ff', type=int, default=256, help='Dimension of the feed-forward layer.')
    parser.add_argument('--e_layers', type=int, default=2, help='Number of encoder layers.')
    parser.add_argument('--d_layers', type=int, default=2, help='Number of decoder layers (if applicable).')
    parser.add_argument('--n_heads', type=int, default=2, help='Number of attention heads (for Transformer-based models).')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
    parser.add_argument('--activation', type=str, default='gelu', help='Activation function.')
    parser.add_argument('--output_attention', action='store_true', help='Whether to output attention weights (for Transformer-based models).')
    # ... (其他来自 Configs 的参数，保持不变) ...
    parser.add_argument('--moving_avg', type=int, default=25, help='Window size for moving average (if used, e.g., in DLinear).')
    parser.add_argument('--seg_len', type=int, default=10, help='Segment length (for SegRNN).')
    parser.add_argument('--channel_independence', type=int, default=1, help='Degree of channel independence (0: channel-mixing, 1: channel-independent).') # Check interpretation
    parser.add_argument('--mixer_kernel_size', type=int, default=8, help='Kernel size for mixer layers (TSMixer).')
    parser.add_argument('--patch_len', type=int, default=10, help='Length of patches (PatchMixer, PatchTST).')
    parser.add_argument('--model_stride', type=int, default=2, help='Stride within the model (e.g., PatchMixer, PatchTST). Name changed to avoid conflict.') # Renamed from stride
    parser.add_argument('--cov_dim', type=int, default=4, help='Dimension of covariates (if used).')
    parser.add_argument('--hidden_dim_cov', type=int, default=16, help='Hidden dimension for covariate processing (if used).')
    parser.add_argument('--decoder_layers', type=int, default=3, help='Number of layers in a specific decoder part (if applicable).')
    parser.add_argument('--embed', type=str, default='timeF', help='Type of embedding (e.g., timeF, fixed, learned).')
    parser.add_argument('--freq', type=str, default='t', help='Frequency for time features (e.g., h, t). Adjusted default.') # Changed default from 'm'
    parser.add_argument('--factor', type=int, default=3, help='Factor for attention mechanisms (e.g., Informer).')
    parser.add_argument('--num_class', type=int, default=10, help='Number of classes (if used for classification tasks within model).') # Usually not for RUL
    parser.add_argument('--dec_in', type=int, default=1, help='Decoder input size (often pred_len for RUL).') # Default to pred_len
    parser.add_argument('--c_out', type=int, default=1, help='Output dimension (usually 1 for RUL).') # Default to 1
    parser.add_argument('--task_name', type=str, default='rul_prediction', help='Task name (e.g., RUL, classification, imputation).')
    parser.add_argument('--CNN_size',type=int, default=64,  help='CNN hidden size.')
    parser.add_argument('--kernel_size',type=int, default=3, help='CNN kernel size.')
    parser.add_argument('--cov_div', type=int, default=4, help='Divisor for covariate processing (if used).')


    args = parser.parse_args()

    # --- 动态调整与验证参数 ---
    # (与上一版本类似，加载 unit/fault 信息并调整 input_size)
    args.dataset = DATASET_LIST[args.dataset_choice]
    args.seq_len = args.window_size
    args.enc_in = args.input_size
    args.c_out = 1 # Ensure output is 1 for RUL
    args.dec_in = args.pred_len # Ensure decoder input matches prediction length

    try:
        if not os.path.exists(args.unit_index_csv): raise FileNotFoundError(f"Unit index file not found: {args.unit_index_csv}")
        if not os.path.exists(args.fault_mode_csv): raise FileNotFoundError(f"Fault mode file not found: {args.fault_mode_csv}")
        unit_index_df = pd.read_csv(args.unit_index_csv)
        df_faults = pd.read_csv(args.fault_mode_csv)
        input_size_add = df_faults.groupby('dataset').size()
        fault_mode_index = df_faults.groupby('dataset')['fault_mode'].apply(list).to_dict()
        dataset_key = args.dataset + '.h5' if not args.dataset.endswith('.h5') else args.dataset
        dataset_row = unit_index_df[unit_index_df['File'] == dataset_key]
        if not dataset_row.empty:
             try:
                 args.index_train = np.fromstring(dataset_row['Dev Units'].values[0].strip('[]'), dtype=np.int8, sep=' ').tolist()
                 args.index_test = np.fromstring(dataset_row['Test Units'].values[0].strip('[]'), dtype=np.int8, sep=' ').tolist()
             except Exception as e: raise ValueError(f"Error parsing unit indices from CSV for {dataset_key}: {e}")
        else: raise ValueError(f"Dataset key '{dataset_key}' not found in 'File' column of {args.unit_index_csv}")
        dataset_num = args.dataset_choice + 1
        args.fault_index = fault_mode_index.get(dataset_num, [])
        if not args.fault_index: print(f"Warning: Fault modes for dataset choice {dataset_num} not found. Using empty list.")
    except (FileNotFoundError, ValueError, Exception) as e:
        print(f"Error processing CSV data: {e}")
        exit(1)

    base_input_size = 20
    timestamp_size = 1
    fault_mode_size = input_size_add.get(dataset_num, 0)
    if fault_mode_size == 0 and args.loadmode in ['cheat1', 'encode|cheat12']: print(f"Warning: Fault mode count for dataset {dataset_num} is 0 or missing.")

    if args.loadmode == 'normal': args.input_size = base_input_size
    elif args.loadmode == 'cheat1':
        args.input_size = base_input_size + fault_mode_size
        args.cov_div = args.cov_div + fault_mode_size
    elif args.loadmode == 'encode|cheat12': args.input_size = base_input_size + timestamp_size + fault_mode_size
    elif args.loadmode == 'normal1': args.input_size = base_input_size + timestamp_size
    elif args.loadmode == 'encode1': args.input_size = base_input_size + timestamp_size
    args.enc_in = args.input_size # Update enc_in again

    # --- 执行主函数 ---
    main(args)

