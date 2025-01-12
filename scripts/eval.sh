#!/bin/bash

lm-eval \
    --model hf \
    --model_args pretrained=../checkpoint/merged_model \
    --tasks mmlu,arithmetic \
    --device cuda:0 \
    --batch_size 4 \
    --output_path eval.log