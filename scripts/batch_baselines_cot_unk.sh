#!/usr/bin/env bash

# Ensure log directory exists
mkdir -p "logs"

# Default variables
model_name_or_path="meta-llama/Llama-3.1-8B-Instruct"
output_dir="./output"
device="cuda"
ckpt_suffix="cot-unk"
max_new_tokens=4096
num_thought_tokens=10
verbose=1

# Argument parsing
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name_or_path) model_name_or_path="$2"; shift 2 ;;
        --max_new_tokens) max_new_tokens="$2"; shift 2 ;;
        --output_dir) output_dir="$2"; shift 2 ;;
        --ckpt_suffix) ckpt_suffix="$2"; shift 2 ;;
        --verbose) verbose="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; shift ;;
    esac
done

# Display basic configuration
echo "=================== [Default Args] ====================="
echo "Model ID: ${model_name_or_path}"
echo "Device: ${device}"
echo "Output dir: ${output_dir}"
echo "Checkpoint suffix: ${ckpt_suffix}"
echo "Max new tokens: ${max_new_tokens}"
echo "Verbose: ${verbose}"
echo "--------------------------------------------------------"

model_name="${model_name_or_path#*/}"

######################### AIME-2024 #########################
dataset="Maxwell-Jia/AIME_2024"
start_time=$(date +%s)

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "Eval Dataset: ${dataset} at $(date)"

cmd="python main.py \
--dataset \"${dataset}\" \
--model_name_or_path \"${model_name_or_path}\" \
--output_dir \"${output_dir}\" \
--device \"${device}\" \
--ckpt_suffix \"${ckpt_suffix}\" \
--max_new_tokens ${max_new_tokens} \
--num_thought_tokens ${num_thought_tokens} \
--max_num_steps 0 \
--verbose ${verbose}"

log_file_name="logs/Baseline-CoT-Unk-AIME2024-${model_name}-max_tokens${max_new_tokens}.log"

# Run the command and redirect output
echo "${cmd} > \"${log_file_name}\""
eval "${cmd} > \"${log_file_name}\""

# Display the script end time
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Evaluation for dataset ${dataset} finished at: $(date)"
echo "Elapsed time: ${elapsed_time} seconds"

######################### AIME-2025 #########################
dataset="opencompass/AIME2025"
start_time=$(date +%s)

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "Eval Dataset: ${dataset} at $(date)"

cmd="python main.py \
--dataset \"${dataset}\" \
--model_name_or_path \"${model_name_or_path}\" \
--output_dir \"${output_dir}\" \
--device \"${device}\" \
--ckpt_suffix \"${ckpt_suffix}\" \
--max_new_tokens ${max_new_tokens} \
--num_thought_tokens ${num_thought_tokens} \
--max_num_steps 0 \
--verbose ${verbose}"

log_file_name="logs/Baseline-CoT-Unk-AIME2025-${model_name}-max_tokens${max_new_tokens}.log"

# Run the command and redirect output
echo "${cmd} > \"${log_file_name}\""
eval "${cmd} > \"${log_file_name}\""

# Display the script end time
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Evaluation for dataset ${dataset} finished at: $(date)"
echo "Elapsed time: ${elapsed_time} seconds"

######################### GSM8K #########################
dataset="openai/gsm8k"
start_time=$(date +%s)

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "Eval Dataset: ${dataset} at $(date)"

cmd="python main.py \
--dataset \"${dataset}\" \
--model_name_or_path \"${model_name_or_path}\" \
--output_dir \"${output_dir}\" \
--device \"${device}\" \
--ckpt_suffix \"${ckpt_suffix}\" \
--max_new_tokens ${max_new_tokens} \
--num_thought_tokens ${num_thought_tokens} \
--max_num_steps 0 \
--verbose ${verbose}"

log_file_name="logs/Baseline-CoT-Unk-GSM8K-${model_name}-max_tokens${max_new_tokens}.log"

# Run the command and redirect output
echo "${cmd} > \"${log_file_name}\""
eval "${cmd} > \"${log_file_name}\""

# Display the script end time
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Evaluation for dataset ${dataset} finished at: $(date)"
echo "Elapsed time: ${elapsed_time} seconds"

######################### MATH-500 #########################
dataset="HuggingFaceH4/MATH-500"
start_time=$(date +%s)

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "Eval Dataset: ${dataset} at $(date)"

cmd="python main.py \
--dataset \"${dataset}\" \
--model_name_or_path \"${model_name_or_path}\" \
--output_dir \"${output_dir}\" \
--device \"${device}\" \
--ckpt_suffix \"${ckpt_suffix}\" \
--max_new_tokens ${max_new_tokens} \
--num_thought_tokens ${num_thought_tokens} \
--max_num_steps 0 \
--verbose ${verbose}"

log_file_name="logs/Baseline-CoT-Unk-MATH500-${model_name}-max_tokens${max_new_tokens}.log"

# Run the command and redirect output
echo "${cmd} > \"${log_file_name}\""
eval "${cmd} > \"${log_file_name}\""

# Display the script end time
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Evaluation for dataset ${dataset} finished at: $(date)"
echo "Elapsed time: ${elapsed_time} seconds"

######################### ASDiv-Aug #########################
dataset="xuyige/ASDiv-Aug"
start_time=$(date +%s)

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "Eval Dataset: ${dataset} at $(date)"

cmd="python main.py \
--dataset \"${dataset}\" \
--model_name_or_path \"${model_name_or_path}\" \
--output_dir \"${output_dir}\" \
--device \"${device}\" \
--ckpt_suffix \"${ckpt_suffix}\" \
--max_new_tokens ${max_new_tokens} \
--num_thought_tokens ${num_thought_tokens} \
--max_num_steps 0 \
--verbose ${verbose}"

log_file_name="logs/Baseline-CoT-Unk-ASDivAug-${model_name}-max_tokens${max_new_tokens}.log"

# Run the command and redirect output
echo "${cmd} > \"${log_file_name}\""
eval "${cmd} > \"${log_file_name}\""

# Display the script end time
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Evaluation for dataset ${dataset} finished at: $(date)"
echo "Elapsed time: ${elapsed_time} seconds"
