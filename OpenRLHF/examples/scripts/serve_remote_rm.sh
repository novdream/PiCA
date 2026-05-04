export CUDA_VISIBLE_DEVICES=2,3
special_token='<|vision_start|>'
set -x

python -m openrlhf.cli.serve_rm \
    --reward_pretrain Qwen2.5-3B-Instruct \
    --port 5000 \
    --param_dtype bf16 \
    --attn_implementation flash_attention_2 \
    --max_len 32768 \
    --batch_size 32 \
    --special_token $special_token \