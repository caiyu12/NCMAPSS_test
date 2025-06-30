#!/bin/bash

# 定义要测试的模型列表
models=("simple" "CNN" "hybrid" "DLinear" "Transformer" "TSMixer") # 示例：只测试几个模型

# 定义要测试的数据集索引列表 (0 到 7)
datasets=(1 2 3 4 5 6)

# 设置循环运行的次数
num_runs=2 # 可以根据需要修改这个值

# 定义要覆盖的参数 (可选)
# 使用 --<参数名> <值> 的格式
# 例如: --epoch 100 --learning_rate 0.0005

# 外层循环：控制整个测试流程的运行次数
for run in $(seq 1 $num_runs); do
  echo "#####################################################"
  echo "Run ${run}/${num_runs}"
  echo "#####################################################"

  # 内层循环：遍历模型和数据集组合
  for model_name in "${models[@]}"; do
    for dataset_idx in "${datasets[@]}"; do
      echo "-----------------------------------------------------"
      echo "Running Test: Run=${run}, Model=${model_name}, Dataset Index=${dataset_idx}"
      echo "-----------------------------------------------------"

      # 构建 python 命令
      python main.py \
        --model "$model_name" \
        --dataset_choice "$dataset_idx" \
        --epoch 150 \
        --batch_size 32 \
        --learning_rate 0.001 \
        --window_size 100 \
        --sampling 10 \
        --load_data_mode "average_in_two" \
        --num_segments 2 \
        --loadmode "cheat1" \
        --device "cuda:0" \
        --log_file "./results/Baseline_log.csv" \
#        --random_seed $((123 + run)) # 使用与运行次数相关的种子

      # 检查上一个命令的退出状态
      if [ $? -ne 0 ]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "ERROR: Failed running Run=${run}, Model=${model_name}, Dataset Index=${dataset_idx}"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        # 可选择退出脚本或继续下一个测试
        # exit 1
      fi
      echo "Finished Test: Run=${run}, Model=${model_name}, Dataset Index=${dataset_idx}"
      echo ""
    done
  done
done

echo "====================================================="
echo "All tests completed."
echo "====================================================="

# 将自己移动到 scripts/ 子文件夹
SCRIPT_NAME=$(basename "$0")
TARGET_DIR="scripts"

# 创建目标目录（如果不存在）
mkdir -p "$TARGET_DIR"

# 移动脚本
#mv "$0" "$TARGET_DIR/$SCRIPT_NAME"

exit 0
