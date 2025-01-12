# alpaca-lora
Use lora to reproduce alpaca.

## Installation
Main dependencies are `openrlhf` and `lm-eval`

### Install openrlhf
Refer to [openrlhf](https://openrlhf.readthedocs.io/en/latest/quick_start.html#installation).

### Install lm-eval
Refer to [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness/).

## Data Generation

The data generation methodology draws inspiration from Alpaca, though with several simplifications to streamline the process. By leveraging the DeepSeek Chat API for generation, we were able to complete the asynchronous requests in approximately 17.5 hours, with the total API costs remaining under $35 USD (approximately 250 CNY).

To generate data, run the following commands:
```python
python generate_instructions_async.py
```

## LoRA Fine-tuning

OpenRLHF is used for easy lora fine-tuning.

```shell
./train_llama3_8b_sft_lora.sh 
```

## Evaluation

lm-evaluation-harness is used for easy evaluation.

```shell
./eval.sh
```