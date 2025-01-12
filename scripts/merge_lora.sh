#!/bin/bash

python merge_lora.py --base_model="meta-llama/Meta-Llama-3-8B" --adapter_path="../checkpoint/llama8b-sft-lora" --output_path="../checkpoint/merged_model"