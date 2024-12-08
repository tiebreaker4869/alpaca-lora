import time
import json
import random
import string
from typing import List, Dict
import re
from rouge_score import rouge_scorer
from openai import OpenAI
import tqdm

class AlpacaDataGenerator:
    def __init__(self, 
                 model_name="deepseek-chat", 
                 temperature=1.0, 
                 top_p=1.0, 
                 base_url="https://api.deepseek.com"):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        self.client = OpenAI(base_url=base_url)
        
        self.system_prompt = """You are a helpful assistant that generates diverse task instructions. These instructions will be used to evaluate language models."""

        self.task_requirements = """Generate diverse task instructions following these requirements:
1. Try not to repeat the verb for each instruction to maximize diversity.
2. The language should be diverse. Combine questions with imperative instructions.
3. Include diverse task types (generation, classification, editing, etc.).
4. Tasks should be completable by a language model (no visual/audio output or real-world actions).
5. Use English and keep instructions to 1-2 sentences.
6. For each instruction, provide:
   - An input (use realistic data, max 100 words)
   - If no specific input needed, use "<noinput>"
   - An appropriate output (max 100 words)

Format each task as:
###
Instruction: [instruction]
Input: [input]
Output: [output]
###"""

    def load_seed_tasks(self, seed_file_path: str) -> List[Dict]:
        """加载种子任务"""
        with open(seed_file_path, 'r') as f:
            seed_tasks = [json.loads(l) for l in f]
        return [
            {
                "instruction": t["instruction"],
                "input": t["instances"][0]["input"],
                "output": t["instances"][0]["output"]
            }
            for t in seed_tasks
        ]

    def create_prompt(self, seed_tasks: List[Dict], num_examples: int = 3) -> List[Dict]:
        """创建聊天格式的提示词"""
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        
        # 添加任务要求
        messages.append({"role": "user", "content": self.task_requirements})
        
        # 添加示例
        examples = random.sample(seed_tasks, num_examples)
        examples_text = "Here are some examples:\n\n"
        
        for _, task in enumerate(examples, 1):
            instruction = re.sub(r"\s+", " ", task["instruction"]).strip().rstrip(":")
            input_text = "<noinput>" if task["input"].lower() == "" else task["input"]
            
            examples_text += f"###\n"
            examples_text += f"Instruction: {instruction}\n"
            examples_text += f"Input: {input_text}\n"
            examples_text += f"Output: {task['output']}\n"
        
        examples_text += "\nNow generate 20 new, diverse task instructions following the same format:"
        messages.append({"role": "user", "content": examples_text})
        
        return messages

    def generate_instructions(self, messages: List[Dict]) -> List[Dict]:
        """使用Chat API生成新指令"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=3072,
                stop=["20.", "20:"]
            )
            return self.parse_response(response)
        except Exception as e:
            print(f"生成指令时出错: {e}")
            return []

    def parse_response(self, response) -> List[Dict]:
        """解析API响应"""
        if not response or not response.choices:
            return []
            
        instructions = []
        raw_text = response.choices[0].message.content
        examples = re.split("###", raw_text)
        
        for example in examples:
            if not example.strip():
                continue
                
            # 解析指令、输入和输出
            parts = example.strip().split("\n")
            current_item = {}
            current_key = None
            
            for part in parts:
                part = part.strip()
                if "Instruction:" in part:
                    current_key = "instruction"
                    current_item[current_key] = part.split("Instruction:")[-1].strip()
                elif "Input:" in part:
                    current_key = "input"
                    current_item[current_key] = part.split("Input:")[-1].strip()
                elif "Output:" in part:
                    current_key = "output"
                    current_item[current_key] = part.split("Output:")[-1].strip()
                elif current_key:
                    current_item[current_key] = current_item.get(current_key, "") + " " + part
            
            if self.validate_instruction(current_item):
                instructions.append(current_item)
                
        return instructions

    def validate_instruction(self, item: Dict) -> bool:
        """验证生成的指令是否有效"""
        if not all(k in item for k in ["instruction", "input", "output"]):
            return False
            
        # 清理和标准化数据
        item["instruction"] = item["instruction"].strip()
        item["input"] = "" if item["input"].lower() == "<noinput>" else item["input"].strip()
        item["output"] = item["output"].strip()
        
        # 基本验证规则
        if len(item["instruction"].split()) <= 3 or len(item["instruction"].split()) > 150:
            return False
            
        # 检查黑名单词
        blacklist = [
            "image", "images", "graph", "graphs", "picture", "pictures",
            "file", "files", "map", "maps", "draw", "plot", "go to",
            "video", "audio", "music", "flowchart", "diagram"
        ]
        if any(word in item["instruction"].lower() for word in blacklist):
            return False
            
        # 检查是否以标点符号开头
        if item["instruction"][0] in string.punctuation:
            return False
            
        # 检查是否以非英文字符开头
        if not item["instruction"][0].isascii():
            return False
            
        return True

    def check_similarity(self, new_instruction: str, existing_instructions: List[str], threshold: float = 0.7) -> bool:
        """检查新指令与现有指令的相似度"""
        if not existing_instructions:
            return True
            
        for existing in existing_instructions:
            # 使用 rougeL 计算相似度
            scores = self.scorer.score(existing, new_instruction)
            if scores['rougeL'].fmeasure > threshold:
                return False
        return True

    def generate_dataset(self, 
                        seed_file: str,
                        num_instructions: int,
                        output_file: str) -> None:
        """生成完整数据集"""
        # 加载种子任务
        seed_tasks = self.load_seed_tasks(seed_file)
        print(f"加载了 {len(seed_tasks)} 个种子任务")
        
        # 初始化数据集
        dataset = []
        existing_instructions = [t["instruction"] for t in seed_tasks]
        
        pbar = tqdm.tqdm(total=num_instructions)
        
        while len(dataset) < num_instructions:
            import pdb; pdb.set_trace()
            
            messages = self.create_prompt(seed_tasks)
            
            new_instructions = self.generate_instructions(messages)
            
            # 过滤相似指令
            for item in new_instructions:
                if self.check_similarity(item["instruction"], existing_instructions):
                    dataset.append(item)
                    existing_instructions.append(item["instruction"])
                    pbar.update(1)
                    
                    # 定期保存
                    if len(dataset) % 100 == 0:
                        self.save_dataset(dataset, output_file)
                        
                if len(dataset) >= num_instructions:
                    break
                    
            # 添加延迟避免超出API限制
            time.sleep(1)
        
        # 最终保存
        self.save_dataset(dataset, output_file)
        print(f"已生成 {len(dataset)} 条指令并保存到 {output_file}")

    def save_dataset(self, dataset: List[Dict], filename: str) -> None:
        """保存数据集"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

def main():
    # 配置参数
    model_name = "deepseek-chat"
    seed_file = "data/seed_tasks.jsonl"
    output_file = "data/alpaca_data.json"
    num_instructions = 52000
    
    # 创建生成器实例
    generator = AlpacaDataGenerator(model_name=model_name)
    
    # 生成数据集
    generator.generate_dataset(
        seed_file=seed_file,
        num_instructions=num_instructions,
        output_file=output_file
    )

if __name__ == "__main__":
    main()