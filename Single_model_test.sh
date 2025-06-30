#!/bin/bash

# 脚本说明：
# 本脚本用于执行 pRNN_sq 模型的两组系列实验。
# 每组实验会针对数据集 0 到 6 进行迭代。
# main.py 脚本应位于上一级目录。

# --- 通用参数 ---
MODEL_NAME="PRISM_RNN"
BASE_EPOCHS=100
TARGET_DEVICE="cuda:2" # 请确保这是 main.py 能识别的有效设备

# 日志文件路径 (相对于 main.py 所在的主目录)
# 例如，如果脚本在 '主目录/scripts/' 中，main.py 在 '主目录/' 中，
# 那么 '../results/debug_log.csv' 将指向 '主目录/results/debug_log.csv'
LOG_FILE_PATH="./results/formal_test.csv"

# 确保日志文件所在的目录存在 (相对于 main.py 的位置)
# mkdir -p "$(dirname "$LOG_FILE_PATH")" # main.py 内部现在会创建，此行可选

echo "统一日志文件将保存在: ${LOG_FILE_PATH}"
echo ""

# --- 实验组 1 ---
echo "--- 开始执行实验组 1 ---"
LR_SET1=0.001
BATCH_SIZE_SET1=32
SAMPLING_RATE_SET1=10
LOAD_DATA_MODE_SET1="average_in_two"
NUM_SEGMENTS_SET1=2 # 新参数
LOAD_MODE_SET1="cheat1"
SEGMENT_LENGTH=10
MODEL_DIMENSION=256    # 对应 --d_model
SEQUENCE_WINDOW_SIZE=50

for dataset_idx in {0..6} # 数据集 0 到 6
do
  echo "实验组 1: 模型=${MODEL_NAME}, 数据集=${dataset_idx}, 学习率=${LR_SET1}"
  python ./main.py \
    --model "${MODEL_NAME}" \
    --dataset_choice "${dataset_idx}" \
    --epoch "${BASE_EPOCHS}" \
    --learning_rate "${LR_SET1}" \
    --log_file "${LOG_FILE_PATH}" \
    --batch_size "${BATCH_SIZE_SET1}" \
    --sampling "${SAMPLING_RATE_SET1}" \
    --load_data_mode "${LOAD_DATA_MODE_SET1}" \
    --num_segments "${NUM_SEGMENTS_SET1}" \
    --loadmode "${LOAD_MODE_SET1}" \
    --device "${TARGET_DEVICE}" \
    --seg_len "${SEGMENT_LENGTH}" \
    --d_model "${MODEL_DIMENSION}" \
    --window_size "${SEQUENCE_WINDOW_SIZE}" \
    # 如果 main.py 需要，请在此处添加 --results_dir, --unit_index_csv 等路径参数
    # 例如: --results_dir "../results" --unit_index_csv "../unit_index.csv"

  if [ $? -ne 0 ]; then
    echo "错误: 实验组 1, 数据集 ${dataset_idx} 执行失败。"
  else
    echo "完成: 实验组 1, 数据集 ${dataset_idx}。"
  fi
  echo "-------------------------------------"
done

echo ""
echo "--- 实验组 1 执行完毕 ---"
echo ""

# --- 实验组 2 ---
echo "--- 开始执行实验组 2 ---"
LR_SET2=0.002
BATCH_SIZE_SET2=1024
SAMPLING_RATE_SET2=100
LOAD_DATA_MODE_SET2="no_average"
NUM_SEGMENTS_SET2=0 # 新参数
LOAD_MODE_SET2="normal"
SEGMENT_LENGTH=5
MODEL_DIMENSION=64    # 对应 --d_model
SEQUENCE_WINDOW_SIZE=50

for dataset_idx in {0..6} # 数据集 0 到 6
do
  echo "实验组 2: 模型=${MODEL_NAME}, 数据集=${dataset_idx}, 学习率=${LR_SET2}"
  python ./main.py \
    --model "${MODEL_NAME}" \
    --dataset_choice "${dataset_idx}" \
    --epoch "${BASE_EPOCHS}" \
    --learning_rate "${LR_SET2}" \
    --log_file "${LOG_FILE_PATH}" \
    --batch_size "${BATCH_SIZE_SET2}" \
    --sampling "${SAMPLING_RATE_SET2}" \
    --load_data_mode "${LOAD_DATA_MODE_SET2}" \
    --num_segments "${NUM_SEGMENTS_SET2}" \
    --loadmode "${LOAD_MODE_SET2}" \
    --device "${TARGET_DEVICE}" \
    --seg_len "${SEGMENT_LENGTH}" \
    --d_model "${MODEL_DIMENSION}" \
    --window_size "${SEQUENCE_WINDOW_SIZE}" \
    # 如果 main.py 需要，请在此处添加 --results_dir, --unit_index_csv 等路径参数

  if [ $? -ne 0 ]; then
    echo "错误: 实验组 2, 数据集 ${dataset_idx} 执行失败。"
  else
    echo "完成: 实验组 2, 数据集 ${dataset_idx}。"
  fi
  echo "-------------------------------------"
done

echo ""
echo "--- 实验组 2 执行完毕 ---"
echo "所有实验已完成。"
