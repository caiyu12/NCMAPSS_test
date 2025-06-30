# 文档：剩余使用寿命（RUL）预测训练脚本

## 1. 概述

该 Python 脚本用于训练和评估多种机器学习模型，以预测 N-CMAPSS 数据集上航空发动机的剩余使用寿命（Remaining Useful Life, RUL）。它自动化了训练流程、超参数配置、模型评估（RMSE 和 NASA 分数）、最佳模型保存以及结果记录等功能。

## 2. 先决条件

* **自定义模块:**
    * `network`: 存放已实现的、无需读入时间戳的时序预测模型。
    * `models`: 存放一部分已实现的、需要使用时间戳的时序预测模型与未实现的模型。
    * `data_process`: 一个 Python 模块 (`data_process.py`)，负责加载、预处理和提供 N-CMAPSS 数据。
* **数据文件:**
    * N-CMAPSS 数据集文件，格式为 HDF5 (`.h5`)。文件名应遵循 `N-CMAPSS_DSXX-YYY.h5` 的模式。
    * `unit_index.csv`: CSV 文件，为每个数据集指定哪些引擎单元用于训练，哪些用于测试。
    * `fault_mode.csv`: CSV 文件，包含有关故障模式的信息，这些信息可能会作为额外的输入特征（尤其是在 `cheat1` 或 `encode|cheat12` 加载模式下）。
    * `model.csv` (可选): CSV 文件，列出了如果要使用 `if __name__ == '__main__':` 块中的循环进行训练，需要训练的模型名称。

## 3. 文件结构（推荐）

```
your_project_directory/
├── dataset                    # 数据集
├── cct_NCMAPSS/               # 主程序文件夹
```

## 4. 配置

主要配置通过 `args_config` 函数和 `Configs` 类（为运行time-series-library移植的文件设置的参数）进行。

* **`args_config(dataset_choice: int, model_name: str) -> Namespace`:**
    * 此函数创建一个 `Namespace` 对象，其中包含大部分训练参数。
    * `dataset_choice`: 一个整数（从 0 开始），选择要使用的 N-CMAPSS 数据集（例如 0 代表 `N-CMAPSS_DS01-005`，1 代表 `N-CMAPSS_DS02-006` 等）。
    * `model_name`: 一个字符串，指定要训练的模型（必须与 `main` 函数中的某个 `elif` 块对应）。
    * **`args_config` 内的关键参数:**
        * `directory`: 项目的根目录。
        * `dataset`: 所选数据集的名称（自动设置）。
        * `epoch`: 训练的总轮数（epochs）。
        * `device`: 用于训练的设备 (`cuda:0` 或 `cpu`)。
        * `max_rul`: RUL 的最大截断值。
        * `input_size`: 模型的输入特征数量。**重要提示：** 此值会根据 `loadmode` 动态调整。
        * `learning_rate`: 优化器的学习率。
        * `batch_size`: 每个批次 (batch) 的样本数量。
        * `model`: 模型的名称（从函数参数传入）。
        * `index_train`, `index_test`: 用于训练和测试的单元 ID 列表（从 `unit_index.csv` 读取）。
        * `stride`, `sampling`, `window_size`, `skip`: 用于从原始数据创建时间窗口序列的参数（由 `Data_Process` 使用）。
        * `load_data_mode`: 加载/预处理数据的特定模式（'average\_in\_two'与'average_in_one'）。
        * `loadmode`: 影响加载哪些数据并调整 `input_size`：
            * `'normal'`: 标准特征（例如 20 个传感器 + 虚拟传感器）。
            * `'cheat1'`: 标准特征 + 来自 `fault_mode.csv` 的故障模式信息。
            * `'encode|cheat12'`: 类似于 'cheat1'，带有额外的时间戳。
        * `fault_index`: 当前数据集的故障模式索引列表（从 `fault_mode.csv` 读取）。
        * `random_seed`: 用于随机数生成器的种子，以确保可复现性。
        * `memory_pinned`: 用于加速数据传输到 GPU 的优化选项。

* **`Configs` 类:**
    * 此类包含特定模型架构（通常是从时间序列库移植过来的，如 Transformer, TSMixer, SegRNN）的超参数。
    * 诸如 `seq_len`, `pred_len`, `d_model`, `e_layers`, `n_heads` 等参数在此定义，并在 `main` 函数中实例化这些特定模型时使用。`configs.enc_in` 会被动态设置为 `args.input_size`。

## 5. 核心组件

* **`Process` 类:**
    * 协调整个训练和测试过程。
    * `__init__`: 初始化数据加载器 (`Data_Process`)、模型、优化器、损失函数，并设置结果保存和日志记录。
    * `Train`: 实现主要的训练循环（遍历 epochs 和 batches），计算损失，执行反向传播，调用 `Test` 进行评估，根据 RMSE 保存最佳模型，并记录结果。
    * `Test`: 在测试数据集上评估当前模型，计算 RMSE 和 NASA PHM 分数。
    * `nasa_score`: 计算 NASA PHM 挑战赛的非对称评估指标。
    * `save_model`/`load_model`: 保存/加载模型权重。
    * `_get_his_best_rmse`: 读取日志文件 (`training_log_new.csv`) 以查找当前模型/数据集组合的历史最佳 RMSE。
    * `_log_to_csv`: 将训练结果（模型名称、最佳 RMSE、时间戳、超参数）写入 CSV 日志文件。
* **`Data_Process` 类 (来自 `data_process.py`)**:
    * 负责加载 `.h5` 文件，进行预处理（归一化、根据 `window_size`, `stride`, `sampling`, `skip` 进行窗口化），处理不同的 `loadmode` 选项，并提供用于训练和测试的 PyTorch DataLoader。
* **模型类 (来自 `network/`)**:
    * 每个类（例如 `SegRNN`, `TSMixer`, `SimpleRULPredictor`）定义了一种用于 RUL 预测的特定神经网络架构。它们继承自 `torch.nn.Module`。
* **`main(dataset_choice, model_name)` 函数:**
    * 设置随机种子。
    * 调用 `args_config` 获取配置。
    * 根据 `model_name` 实例化选定的模型，并在需要时传递 `configs`。
    * 创建 `Process` 类的实例。
    * 通过调用 `instance.Train()` 启动训练过程。

## 6. 如何运行脚本

1.  **设置结构:** 确保所有必需的文件 (`.py`, `.h5`, `.csv`) 和目录 (`network/`) 都按照第 3 节所述的方式放置。
2.  **安装依赖:** (确保已安装 Python 和 PyTorch)
3.  **调整代码中的配置:**
    * 找到脚本末尾的 `if __name__ == '__main__':` 块。
    * **对于单个实验:** 修改 `main(0, 'simple')` 这一行。将 `0` 替换为您想要的数据集的索引（0 代表 DS01，1 代表 DS02，...），并将 `'simple'` 替换为您想要训练的模型的名称（例如 `'SegRNN'`, `'TSMixer'`, `'DLinear'`)。
    * **对于多个实验:** 取消注释循环代码并根据需要进行调整。如果使用 `df.iloc[j]['model']` 的循环，请确保 `model.csv` 文件存在并包含所需的模型名称。
4.  **运行脚本:** 打开终端或命令提示符，导航到项目目录，然后运行脚本：
    ```bash
    python main_script.py
    ```
  或直接运行main.py(main_encode原为encoder相关模型设计，在数据中加入时间戳），在运行前应先扩展Linux文件限制：
  ```bash
  unlimit -n 65535
  ```

## 7. 输出

* **控制台输出:** 在训练期间，将为每个 epoch 输出信息（epoch 编号、训练损失、测试集上的 RMSE 和分数）。关于新的最佳 RMSE 和模型保存的消息也会显示。
* **`results/` 目录:**
    * `training_log_new.csv`: 一个 CSV 文件，每行记录一次完整训练运行的结果（模型名称、最佳 RMSE、训练时间、数据集、超参数）。
    * `results/<model_name>/`: 为每个训练的模型创建一个子目录。
        * `results/<model_name>/<model_name>_best_temp_in_dsXX.pth`: 临时文件，保存当前运行中 RMSE 最佳的模型。
        * `results/<model_name>/<model_name>_best_in_dsXX.pth`: 该数据集的最终最佳模型文件。仅当当前运行中获得的 RMSE 优于此模型/数据集组合的历史最佳 RMSE（从 `training_log_new.csv` 读取）时，才会创建/覆盖此文件。
## 8. 已实现模型
1. model.csv中模型。
2. PatchTST, Timemixer, [Transformer_encode](https://ieeexplore.ieee.org/document/9864208) (encoder模型无一例外均表现的很差，可能在模型设置上出了大问题）（update：调整d_model后模型表现大幅提升）
attn_mask（一般来说，应用滑窗操作后无需使用因果掩码）使用如下方式实现：
```
mask = torch.triu(torch.ones((size, size), device=device), diagonal=1).bool()
```
## 9. 添加新模型

1.  **创建模型定义:** 在 `network/` 目录下创建一个新的 Python 文件（例如 `MyNewModel.py`）。在其中定义您的模型类，使其继承自 `torch.nn.Module`。实现 `__init__` 和 `forward` 方法。
2.  **导入模型:** 在主脚本 (`main_script.py`) 的开头添加导入语句以导入您的新模型类：`from network.MyNewModel import MyNewModel`。
3.  **在 `main` 中实例化:** 在 `main` 函数中添加一个新的 `elif` 块，以便在传递模型名称作为 `model_name` 时实例化您的模型：
    ```python
    elif args.model == 'MyNewModel':
        model = MyNewModel(
            # 在此传递必要的参数,
            # 例如 args.input_size, args.window_size 等
            # 或 configs 中的参数 (如果适用)
        )
    ```
4.  **配置 (可选):** 如果您的模型需要 `args` 或 `Configs` 中没有的特定超参数，请扩展其中一个类或在实例化时直接传递值。
5.  **开始训练:** 按照第 6 节所述运行脚本，并将您的新模型名称 (`'MyNewModel'`) 传递给 `main` 函数。
