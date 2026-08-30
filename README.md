# Initial implementation of NCMAPSS dataset RUL prediction.
## Dataset
We chose the N-CMAPSS dataset as the validation dataset of our RUL prediction model. The dataset can be found at https://data.phmsociety.org/2021-phm-conference-data-challenge/.

We processed the data by applying the method mentioned in [Domain Adaptive Remaining Useful Life Prediction with Transformer](https://ieeexplore.ieee.org/document/9864208).

## Start Here: PRISM-RNN

The main contribution of this repository is **PRISM-RNN (Prognostic Routing of Independent Strategy Modules)**.

| File | Description |
|---|---|
| [`network/PRISM_RNN.py`](network/PRISM_RNN.py) | Main PRISM-RNN architecture |
| [`layers/PRISM_layer.py`](layers/PRISM_layer.py) | GMM routing gate, normalization, decomposition, and frequency-domain modules |
| [`main.py`](main.py) | Training and evaluation entry point |
| [`data_process.py`](data_process.py) | N-CMAPSS preprocessing and DataLoader construction |
| [`unit_index.csv`](unit_index.csv) | Training/test engine-unit split |
| [`fault_mode.csv`](fault_mode.csv) | Fault-mode definitions |

PRISM-RNN accepts an input tensor with shape:

```text
[batch_size, sequence_length, number_of_features]
```

and predicts one RUL value for each input sequence.

Its main workflow is:

1. Extract channel-level statistical features and use a GMM gate to identify homogeneous or heterogeneous sensor behaviour.
2. Route the input through channel-dependent and channel-independent GRU branches.
3. Apply frequency-domain channel processing, trend/seasonal decomposition, and final RUL projection.

## Quick Start

The following command performs a one-epoch PRISM-RNN run on DS01 using the CPU:

```bash
python main.py \
  --model PRISM_RNN \
  --dataset_choice 0 \
  --device cpu \
  --epoch 1 \
  --batch_size 32 \
  --window_size 50 \
  --seg_len 10 \
  --d_model 64 \
  --num_segments 2 \
  --loadmode cheat1 \
  --load_data_mode average_in_two \
  --random_seed 42
```

`window_size` must be divisible by `seg_len`.

For GPU training, replace `--device cpu` with an available device, for example:

```bash
--device cuda:0
```

A successful run prints the selected model, dataset, number of trainable parameters, training loss, validation RMSE, and NASA score.

## Main Experiment Configuration

A typical PRISM-RNN experiment can be started with:

```bash
python main.py \
  --model PRISM_RNN \
  --dataset_choice 0 \
  --device cuda:0 \
  --epoch 100 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --window_size 50 \
  --seg_len 10 \
  --d_model 256 \
  --sampling 10 \
  --num_segments 2 \
  --loadmode cheat1 \
  --load_data_mode average_in_two \
  --random_seed 42
```

Important parameters:

| Parameter | Meaning |
|---|---|
| `--model` | Model name; use `PRISM_RNN` for the proposed model |
| `--dataset_choice` | N-CMAPSS subset index |
| `--window_size` | Input sequence length |
| `--seg_len` | Length of each recurrent segment |
| `--d_model` | Hidden representation dimension |
| `--num_segments` | Number of averaged segments during preprocessing |
| `--loadmode` | Input-feature configuration |
| `--random_seed` | Random seed for reproducibility |

## Outputs

After training, the program produces:

```text
results/
├── all_experiments_log.csv
└── <dataset>/
    └── PRISM_RNN/
        └── PRISM_RNN_final_seed<seed>.pth

visualization/
└── <dataset>/
    └── PRISM_RNN/
        └── *.png
```

- `all_experiments_log.csv` records the hyperparameters, RMSE, and NASA score.
- The `.pth` file contains the best model weights.
- The visualization compares predicted and ground-truth RUL.


