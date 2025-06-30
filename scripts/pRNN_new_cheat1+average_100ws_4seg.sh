#!/bin/bash

# 定义要测试的模型列表
models=("poolRNN" "pool_pRNN" "p2pRNN" "pRNN_sq") # 示例：只测试几个模型

# 定义要测试的数据集索引列表 (0 到 7)
datasets=(1 2 3 4 5 6)

# 设置主程序重复运行的总次数
num_runs=2

# 循环测试每个模型和数据集组合
for run in $(seq 1 $num_runs); do
  echo "#####################################################"
  echo "Run ${run}/${num_runs}"
  echo "#####################################################"

  for model_name in "${models[@]}"; do
    for dataset_idx in "${datasets[@]}"; do
      echo "-----------------------------------------------------"
      echo "Running Test: Run=${run}, Model=${model_name}, Dataset Index=${dataset_idx}"
      echo "-----------------------------------------------------"

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
        --log_file "./results/pRNN_new_log.csv" \
#        --random_seed $((123 + run))  # 每次运行使用不同的种子

      if [ $? -ne 0 ]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "ERROR: Failed running Run=${run}, Model=${model_name}, Dataset Index=${dataset_idx}"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
      fi

      echo "Finished Test: Run=${run}, Model=${model_name}, Dataset Index=${dataset_idx}"
      echo ""
    done
  done
done

echo "====================================================="
echo "All tests completed."
echo "====================================================="

exit 0
