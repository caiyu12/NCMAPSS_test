#!/bin/bash

# 定义要测试的模型列表
# 可以从 model.csv 读取，或者直接在此处定义
# models=("TSMixer" "simple" "CNN" "hybrid" "DLinear" "Transformer" "SegRNN" "pRNN" "TC_pRNN")
models=("SegRNN" "TC_pRNN" "SA_pRNN" "simple" "CA_cov_SegRNN" "SA_CA_SegRNN" "SA_CA_pRNN" "SA_SegRNN") # 示例：只测试几个模型

# 定义要测试的数据集索引列表 (0 到 7)
datasets=(0 1 2 3)


# 定义要覆盖的参数 (可选)
# 使用 --<参数名> <值> 的格式
# 例如: --epoch 100 --learning_rate 0.0005

# 循环测试每个模型和数据集组合
for model_name in "${models[@]}"; do
  for dataset_idx in "${datasets[@]}"; do
    echo "-----------------------------------------------------"
    echo "Running Test: Model=${model_name}, Dataset Index=${dataset_idx}"
    echo "-----------------------------------------------------"

    # 构建 python 命令
    # 将所有需要的参数传递给 main.py
    # 使用默认值的参数不需要显式传递
    python main.py \
      --model "$model_name" \
      --dataset_choice "$dataset_idx" \
      --epoch 150 \
      --batch_size 32 \
      --learning_rate 0.001 \
      --window_size 100 \
      --sampling 10 \
      --load_data_mode "average_in_four" \
      --num_segments 4 \
      --loadmode "cheat1" \
      --device "cuda:2" \
      --log_file "./results/cheat1_aver_log1.csv" \
#      --random_seed 123 # 固定种子以便比较

    # 检查上一个命令的退出状态
    if [ $? -ne 0 ]; then
      echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
      echo "ERROR: Failed running Model=${model_name}, Dataset Index=${dataset_idx}"
      echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
      #可以选择退出脚本或继续下一个测试
      # exit 1
    fi
    echo "Finished Test: Model=${model_name}, Dataset Index=${dataset_idx}"
    echo ""
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
mv "$0" "$TARGET_DIR/$SCRIPT_NAME"

exit 0