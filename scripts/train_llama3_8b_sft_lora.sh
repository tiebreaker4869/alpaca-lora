set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
    --max_len 512 \
    --dataset json@../data \
    --input_key instruction \
    --output_key output \
    --train_batch_size 64 \
    --micro_train_batch_size 4 \
    --max_samples 100000 \
    --max_ckpt_num 15 \
    --pretrain meta-llama/Meta-Llama-3-8B \
    --save_path ../checkpoint/llama8b-sft-lora \
    --save_steps 200 \
    --logging_steps 100 \
    --eval_steps 200 \
    --zero_stage 2 \
    --max_epochs 3 \
    --bf16 \
    --load_checkpoint \
    --gradient_checkpointing \
    --flash_attn \
    --packing_samples \
    --learning_rate 5e-5 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --use_wandb b4fa4df21f5a80ff855a3802563231645b5e590e \
    --wandb_project playground \
    --wandb_run_name llama8b-sft-lora-exp
EOF

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi