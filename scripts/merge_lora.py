import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_peft_adapter(base_model_name, adapter_path, output_path):
    """
    合并基础模型和PEFT adapter
    
    参数:
        base_model_name: 基础模型名称或路径
        adapter_path: PEFT adapter路径
        output_path: 合并后模型的保存路径
    """
    print(f"正在加载基础模型: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    
    print(f"正在加载adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("正在合并权重...")
    # 获取合并后的模型
    merged_model = model.merge_and_unload()
    
    print(f"正在保存合并后的模型到: {output_path}")
    # 直接保存合并后的模型，而不是adapter
    merged_model.save_pretrained(output_path)
    
    # 保存tokenizer
    print("正在保存tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(output_path)
    
    print("合并完成!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True, help="Base model name or path")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to PEFT adapter")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save merged model")
    args = parser.parse_args()
    
    merge_peft_adapter(args.base_model, args.adapter_path, args.output_path)