model=$1
dataset=$2
attr=$3
config=$4
gpu_ids=$5
seed=$6

for ((i=0; i<=4; i++))
do
python train.py --model ${model}\
  --dataset_mode ${dataset}\
  --pred_attr ${attr}\
  --enable_val\
  --gpu_ids ${gpu_ids}\
  --config ${config}\
  --save_best\
  --seed $((${seed}+${i}))\
  --eval_epoch_freq 1\
  --num_train_target 3\
  --num_threads 4
done
