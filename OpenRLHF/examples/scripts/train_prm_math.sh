export CUDA_VISIBLE_DEVICES=4,5
set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_prm \
   --save_path Qwen2.5-3B-Instruct-prm \
   --save_steps 500 \
   --logging_steps 1 \
   --eval_steps 100 \
   --train_batch_size 256 \
   --micro_train_batch_size 32 \
   --pretrain Qwen2.5-3B-Instruct  \
   --param_dtype bf16 \
   --neg_weight 1.0 \
   --max_epochs 1 \
   --max_len 8192 \
   --max_samples 500000\
   --zero_stage 3 \
   --learning_rate 5e-6 \
   --dataset zhuzilin/Math-Shepherd  \
   --input_key input \
   --label_key value \
   --attn_implementation flash_attention_2 \
   --load_checkpoint \
   --gradient_checkpointing \
   --packing_samples \
   --wandb_group search_prm \
   --placeholder_token ки \
   --reward_tokens + -\
   --use_wandb True \
   --wandb_run_name prm_math
EOF
     # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
     # --packing_samples
     #    


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
