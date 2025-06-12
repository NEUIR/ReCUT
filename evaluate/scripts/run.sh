export CUDA_VISIBLE_DEVICES=3
nohup python /mnt1/wumingyan/lixinze/Data/dpo_train/Search-o1/scripts/run_direct_gen.py \
    --dataset_name aime \
    --split test \
    --model_short_name mergekit_long_error_long_true_2_700_4_400_sce_75 \
    --model_path /mnt1/wumingyan/lixinze/merge_model/mergekit_long_error_long_true_2_700_4_400_sce_75 > /mnt1/wumingyan/lixinze/Data/dpo_train/Search-o1/test_log/aime_raw.log 2>&1 &

python /mnt1/wumingyan/lixinze/Data/dpo_train/Search-o1/scripts/run_direct_gen.py \
    --dataset_name aime25 \
    --split test \
    --model_short_name mergekit_long_error_long_true_2_700_4_400_sce_75 \
    --model_path /mnt1/wumingyan/lixinze/merge_model/mergekit_long_error_long_true_2_700_4_400_sce_75


export CUDA_VISIBLE_DEVICES=1
python /mnt1/wumingyan/lixinze/Data/dpo_train/Search-o1/scripts/run_direct_gen.py \
    --dataset_name amc \
    --split test \
    --model_short_name long-tru-200-check \
    --output_dir /mnt1/wumingyan/lixinze/Data/dpo_train/Search-o1/outputs/llama3.1 \
    --model_path /mnt1/wumingyan/lixinze/checkpoint/dpo_checkpoint/llama_true_processed_dataset_1_most_long_check_1e5/200-check


export CUDA_VISIBLE_DEVICES=2
python /mnt1/wumingyan/lixinze/Data/dpo_train/Search-o1/scripts/run_direct_gen.py \
    --dataset_name math500 \
    --split test \
    --model_short_name long-tru-200-check \
    --output_dir /mnt1/wumingyan/lixinze/Data/dpo_train/Search-o1/outputs/llama3.1 \
    --model_path /mnt1/wumingyan/lixinze/checkpoint/dpo_checkpoint/llama_true_processed_dataset_1_most_long_check_1e5/200-check

export CUDA_VISIBLE_DEVICES=3
python /mnt1/wumingyan/lixinze/Data/dpo_train/Search-o1/scripts/run_direct_gen.py \
    --dataset_name aime \
    --split test \
    --model_short_name long-tru-200-check \
    --output_dir /mnt1/wumingyan/lixinze/Data/dpo_train/Search-o1/outputs/llama3.1 \
    --model_path /mnt1/wumingyan/lixinze/checkpoint/dpo_checkpoint/llama_true_processed_dataset_1_most_long_check_1e5/200-check

export CUDA_VISIBLE_DEVICES=0
python /mnt1/wumingyan/lixinze/Data/dpo_train/Search-o1/scripts/run_direct_gen.py \
    --dataset_name aime25 \
    --split test \
    --model_short_name long-tru-200-check \
    --output_dir /mnt1/wumingyan/lixinze/Data/dpo_train/Search-o1/outputs/llama3.1 \
    --model_path /mnt1/wumingyan/lixinze/checkpoint/dpo_checkpoint/llama_true_processed_dataset_1_most_long_check_1e5/200-check

export CUDA_VISIBLE_DEVICES=4
python /mnt1/wumingyan/lixinze/Data/dpo_train/Search-o1/scripts/run_direct_gen.py \
    --dataset_name gsm8k \
    --split test \
    --model_short_name long-tru-200-check \
    --output_dir /mnt1/wumingyan/lixinze/Data/dpo_train/Search-o1/outputs/llama3.1 \
    --model_path /mnt1/wumingyan/lixinze/checkpoint/dpo_checkpoint/llama_true_processed_dataset_1_most_long_check_1e5/200-check

export CUDA_VISIBLE_DEVICES=5
python /mnt1/wumingyan/lixinze/Data/dpo_train/Search-o1/scripts/run_direct_gen.py \
    --dataset_name olympiad \
    --split test \
    --model_short_name long-tru-100-check \
    --output_dir /mnt1/wumingyan/lixinze/Data/dpo_train/Search-o1/outputs/llama3.1 \
    --model_path /mnt1/wumingyan/lixinze/checkpoint/dpo_checkpoint/llama_true_processed_dataset_1_most_long_check_1e5/100-check


export CUDA_VISIBLE_DEVICES=5
python /mnt1/wumingyan/lixinze/Data/dpo_train/Search-o1/scripts/run_direct_gen.py \
    --dataset_name minervamath \
    --split test \
    --model_short_name long-tru-200-check \
    --output_dir /mnt1/wumingyan/lixinze/Data/dpo_train/Search-o1/outputs/llama3.1 \
    --model_path /mnt1/wumingyan/lixinze/checkpoint/dpo_checkpoint/llama_true_processed_dataset_1_most_long_check_1e5/200-check


export CUDA_VISIBLE_DEVICES=1
nohup python scripts/run_direct_gen.py \
    --dataset_name amc \
    --split test \
    --model_path /mnt1/wumingyan/lixinze/pretrain_model > /mnt1/wumingyan/lixinze/Data/dpo_train/Search-o1/test_log/amc_raw.log 2>&1 &


export CUDA_VISIBLE_DEVICES=0
nohup python scripts/run_direct_gen.py \
    --dataset_name math500 \
    --split test \
    --subset_num 50 \
    --model_path /mnt1/wumingyan/lixinze/pretrain_model > /mnt1/wumingyan/lixinze/Data/dpo_train/Search-o1/test_log/math500_50_raw.log 2>&1 &

export CUDA_VISIBLE_DEVICES=0
nohup python scripts/run_direct_gen.py \
    --dataset_name aime \
    --split test \
    --model_path /mnt1/wumingyan/lixinze/checkpoint/dpo_checkpoint/700-check > /mnt1/wumingyan/lixinze/Data/dpo_train/Search-o1/test_log/aime_dpo-700.log 2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python scripts/run_direct_gen.py \
    --dataset_name amc \
    --split test \
    --model_path /mnt1/wumingyan/lixinze/checkpoint/dpo_checkpoint/700-check > /mnt1/wumingyan/lixinze/Data/dpo_train/Search-o1/test_log/amc_dpo-700.log 2>&1 &

export CUDA_VISIBLE_DEVICES=2
nohup python scripts/run_direct_gen.py \
    --dataset_name math500 \
    --split test \
    --subset_num 50 \
    --model_path /mnt1/wumingyan/lixinze/checkpoint/dpo_checkpoint/200-check > /mnt1/wumingyan/lixinze/Data/dpo_train/Search-o1/test_log/math500_50_dpo-200.log 2>&1 &

export CUDA_VISIBLE_DEVICES=0
nohup python scripts/run_direct_gen.py \
    --dataset_name aime \
    --split test \
    --model_path /mnt1/wumingyan/lixinze/checkpoint/dpo_checkpoint/200-check > /mnt1/wumingyan/lixinze/Data/dpo_train/Search-o1/test_log/aime_dpo-200.log 2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python scripts/run_direct_gen.py \
    --dataset_name amc \
    --split test \
    --model_path /mnt1/wumingyan/lixinze/checkpoint/dpo_checkpoint/200-check > /mnt1/wumingyan/lixinze/Data/dpo_train/Search-o1/test_log/amc_dpo-200.log 2>&1 &
