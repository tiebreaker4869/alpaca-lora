# alpaca-lora
Use lora to reproduce alpaca.

## Installation
TBA

## Data Generation

The data generation methodology draws inspiration from Alpaca, though with several simplifications to streamline the process. By leveraging the DeepSeek Chat API for generation, we were able to complete the asynchronous requests in approximately 17.5 hours, with the total API costs remaining under $35 USD (approximately 250 CNY)

To generate data, run the following commands:
```python
python generate_instructions_async.py
```

## LoRA Finetuning

LLaMA-Factory is used for easy lora finetuning.