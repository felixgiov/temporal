#!/bin/bash
LR=$1
GRAD=$2
COMB=$3
NUM=$4
GPU=$5
TASK2=false
TASK3=false
TASK4=false
TASK5=false

if [ COMB = "12" ]
then
  TASK2=true
elif [ COMB = "13" ]
then
  TASK3=true
elif [ COMB = "14" ]
then
  TASK4=true
elif [ COMB = "15" ]
then
  TASK5=true
fi

cd projects/research
source venv/bin/activate
export CUDA_VISIBLE_DEVICES="$GPU"
python multi_run.py --data_dir ./datasets \
--train_data /home/felix/projects/research/datasets/MCTACO/cross_val/train_"$NUM".tsv \
--dev_data /home/felix/projects/research/datasets/MCTACO/cross_val/dev_"$NUM".tsv \
--labels ./datasets/TBAQ-cleaned/tbaq_class_timex_labels.txt \
--model_name_or_path roberta-large \
--output_dir ./multi_results_"$LR"e-5_8x"$GRAD"_"$NUM"/"$COMB" \
--learning_rate "$LR"e-5 \
--max_seq_length 128 \
--num_train_epochs 20 \
--per_gpu_train_batch_size 8 \
--per_gpu_eval_batch_size 16 \
--gradient_accumulation_steps "$GRAD" \
--save_steps 50000 \
--seed 32 \
--train_matres "$TASK4" \
--train_event "$TASK3" \
--train_timex "$TASK2" \
--train_duration "$TASK5" \
--do_train true \
--do_eval false \
--do_predict true \
--overwrite_output_dir true