#!/bin/bash
LR=$1
GRAD=$2
GPU=$3

cd projects/research
source venv/bin/activate
export CUDA_VISIBLE_DEVICES="$GPU"
python multi_run.py --data_dir ./datasets \
--train_data /home/felix/projects/research/datasets/MCTACO/cross_val/train_0.tsv \
--dev_data /home/felix/projects/research/datasets/MCTACO/cross_val/dev_0.tsv \
--labels ./datasets/TBAQ-cleaned/tbaq_class_timex_labels.txt \
--model_name_or_path ./multi_results/multi_results_seed31_"$LR"e-5_8x"$GRAD"_0/12_ensemble \
--model_name_or_path_2 ./multi_results/multi_results_seed31_"$LR"e-5_8x"$GRAD"_1/12_ensemble \
--model_name_or_path_3 ./multi_results/multi_results_seed31_"$LR"e-5_8x"$GRAD"_2/12_ensemble \
--model_name_or_path_4 ./multi_results/multi_results_seed31_"$LR"e-5_8x"$GRAD"_3/12_ensemble \
--model_name_or_path_5 ./multi_results/multi_results_seed31_"$LR"e-5_8x"$GRAD"_4/12_ensemble \
--model_name_or_path_6 ./multi_results/multi_results_seed32_"$LR"e-5_8x"$GRAD"_0/12_ensemble \
--model_name_or_path_7 ./multi_results/multi_results_seed32_"$LR"e-5_8x"$GRAD"_1/12_ensemble \
--model_name_or_path_8 ./multi_results/multi_results_seed32_"$LR"e-5_8x"$GRAD"_2/12_ensemble \
--model_name_or_path_9 ./multi_results/multi_results_seed32_"$LR"e-5_8x"$GRAD"_3/12_ensemble \
--model_name_or_path_10 ./multi_results/multi_results_seed33_"$LR"e-5_8x"$GRAD"_4/12_ensemble \
--model_name_or_path_11 ./multi_results/multi_results_seed33_"$LR"e-5_8x"$GRAD"_0/12_ensemble \
--model_name_or_path_12 ./multi_results/multi_results_seed33_"$LR"e-5_8x"$GRAD"_1/12_ensemble \
--model_name_or_path_13 ./multi_results/multi_results_seed33_"$LR"e-5_8x"$GRAD"_2/12_ensemble \
--model_name_or_path_14 ./multi_results/multi_results_seed33_"$LR"e-5_8x"$GRAD"_3/12_ensemble \
--model_name_or_path_15 ./multi_results/multi_results_seed33_"$LR"e-5_8x"$GRAD"_4/12_ensemble \
--model_name_or_path_16 ./multi_results/multi_results_seed31_"$LR"e-5_8x"$GRAD"_0/14_ensemble \
--model_name_or_path_17 ./multi_results/multi_results_seed31_"$LR"e-5_8x"$GRAD"_1/14_ensemble \
--model_name_or_path_18 ./multi_results/multi_results_seed31_"$LR"e-5_8x"$GRAD"_2/14_ensemble \
--model_name_or_path_19 ./multi_results/multi_results_seed31_"$LR"e-5_8x"$GRAD"_3/14_ensemble \
--model_name_or_path_20 ./multi_results/multi_results_seed31_"$LR"e-5_8x"$GRAD"_4/14_ensemble \
--model_name_or_path_21 ./multi_results/multi_results_seed32_"$LR"e-5_8x"$GRAD"_0/14_ensemble \
--model_name_or_path_22 ./multi_results/multi_results_seed32_"$LR"e-5_8x"$GRAD"_1/14_ensemble \
--model_name_or_path_23 ./multi_results/multi_results_seed32_"$LR"e-5_8x"$GRAD"_2/14_ensemble \
--model_name_or_path_24 ./multi_results/multi_results_seed32_"$LR"e-5_8x"$GRAD"_3/14_ensemble \
--model_name_or_path_25 ./multi_results/multi_results_seed33_"$LR"e-5_8x"$GRAD"_4/14_ensemble \
--model_name_or_path_26 ./multi_results/multi_results_seed33_"$LR"e-5_8x"$GRAD"_0/14_ensemble \
--model_name_or_path_27 ./multi_results/multi_results_seed33_"$LR"e-5_8x"$GRAD"_1/14_ensemble \
--model_name_or_path_28 ./multi_results/multi_results_seed33_"$LR"e-5_8x"$GRAD"_2/14_ensemble \
--model_name_or_path_29 ./multi_results/multi_results_seed33_"$LR"e-5_8x"$GRAD"_3/14_ensemble \
--model_name_or_path_30 ./multi_results/multi_results_seed33_"$LR"e-5_8x"$GRAD"_4/14_ensemble \
--output_dir ./multi_results/multi_results_seed_"$LR"e-5_8x"$GRAD"_ensemble/12_14 \
--learning_rate "$LR"e-5 \
--max_seq_length 128 \
--num_train_epochs 20 \
--per_gpu_train_batch_size 8 \
--per_gpu_eval_batch_size 16 \
--gradient_accumulation_steps "$GRAD" \
--save_steps 50000 \
--seed 31 \
--do_predict \
--overwrite_output_dir

##!/bin/bash
#SEED=$1
#LR=$2
#GRAD=$3
#GPU=$4
#
#cd projects/research
#source venv/bin/activate
#export CUDA_VISIBLE_DEVICES="$GPU"
#python multi_run_ensemble_fifteen.py --data_dir ./datasets \
#--train_data /home/felix/projects/research/datasets/MCTACO/cross_val/train_0.tsv \
#--dev_data /home/felix/projects/research/datasets/MCTACO/cross_val/dev_0.tsv \
#--labels ./datasets/TBAQ-cleaned/tbaq_class_timex_labels.txt \
#--model_name_or_path /orange/felix/multi_results/multi_run_cosmos_new_2e_default_dropout/multi_results_seed33_1e-5_8x4/1/10 \
#--model_name_or_path_2 /orange/felix/multi_results/multi_run_cosmos_new_2e_default_dropout/multi_results_seed10321_1e-5_8x4/1/10 \
#--model_name_or_path_3 /orange/felix/multi_results/multi_run_cosmos_new_2e_default_dropout/multi_results_seed76567_1e-5_8x4/1/10 \
#--model_name_or_path_4 /orange/felix/multi_results/multi_run_cosmos_new_2e_default_dropout/multi_results_seed33_1e-5_8x8/12/10 \
#--model_name_or_path_5 /orange/felix/multi_results/multi_run_cosmos_new_2e_default_dropout/multi_results_seed10321_1e-5_8x8/12/10 \
#--model_name_or_path_6 /orange/felix/multi_results/multi_run_cosmos_new_2e_default_dropout/multi_results_seed76567_1e-5_8x8/12/10 \
#--model_name_or_path_7 /orange/felix/multi_results/multi_run_cosmos_new_2e_default_dropout/multi_results_seed33_1e-5_8x2/13/10 \
#--model_name_or_path_8 /orange/felix/multi_results/multi_run_cosmos_new_2e_default_dropout/multi_results_seed10321_1e-5_8x2/13/10 \
#--model_name_or_path_9 /orange/felix/multi_results/multi_run_cosmos_new_2e_default_dropout/multi_results_seed76567_1e-5_8x2/13/10 \
#--model_name_or_path_10 /orange/felix/multi_results/multi_run_cosmos_new_2e_default_dropout/multi_results_seed33_1e-5_8x8/14/9 \
#--model_name_or_path_11 /orange/felix/multi_results/multi_run_cosmos_new_2e_default_dropout/multi_results_seed10321_1e-5_8x8/14/9 \
#--model_name_or_path_12 /orange/felix/multi_results/multi_run_cosmos_new_2e_default_dropout/multi_results_seed76567_1e-5_8x8/14/9 \
#--model_name_or_path_13 /orange/felix/multi_results/multi_run_cosmos_new_2e_default_dropout/multi_results_seed33_1e-5_8x4/15/9 \
#--model_name_or_path_14 /orange/felix/multi_results/multi_run_cosmos_new_2e_default_dropout/multi_results_seed10321_1e-5_8x4/15/9 \
#--model_name_or_path_15 /orange/felix/multi_results/multi_run_cosmos_new_2e_default_dropout/multi_results_seed76567_1e-5_8x4/15/9 \
#--output_dir /orange/felix/multi_results/multi_run_cosmos_new_2e_default_dropout/_result_ensemble \
#--learning_rate "$LR"e-5 \
#--max_seq_length 128 \
#--num_train_epochs 20 \
#--per_gpu_train_batch_size 8 \
#--per_gpu_eval_batch_size 16 \
#--gradient_accumulation_steps "$GRAD" \
#--save_steps 50000 \
#--seed "$SEED" \
#--do_predict \
#--overwrite_output_dir


