import time
import json
import random
import string
import asyncio
import concurrent.futures
from typing import List, Dict
import re
from rouge_score import rouge_scorer
from openai import AsyncOpenAI
import tqdm
from collections import deque
from functools import lru_cache

class AlpacaDataGenerator:
    def __init__(self, 
                 model_name="deepseek-chat", 
                 temperature=1.0, 
                 top_p=1.0, 
                 base_url="https://api.deepseek.com",
                 batch_size=5,
                 max_workers=3):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        self.client = AsyncOpenAI(base_url=base_url)
        
        # Cache recently checked instructions to avoid repeated Rouge calculations
        self.recent_instructions = deque(maxlen=1000)
        
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

    @lru_cache(maxsize=1000)
    def load_seed_tasks(self, seed_file_path: str) -> tuple:
        """加载种子任务并缓存结果"""
        with open(seed_file_path, 'r') as f:
            seed_tasks = [json.loads(l) for l in f]
        return tuple(
            {
                "instruction": t["instruction"],
                "input": t["instances"][0]["input"],
                "output": t["instances"][0]["output"]
            }
            for t in seed_tasks
        )

    def create_prompts_batch(self, seed_tasks: tuple, batch_size: int, num_examples: int = 3) -> List[List[Dict]]:
        """批量创建提示词"""
        prompts = []
        for _ in range(batch_size):
            messages = [
                {"role": "system", "content": self.system_prompt},
            ]
            
            messages.append({"role": "user", "content": self.task_requirements})
            
            examples = random.sample(seed_tasks, num_examples)
            examples_text = "Here are some examples:\n\n"
            
            for task in examples:
                instruction = re.sub(r"\s+", " ", task["instruction"]).strip().rstrip(":")
                input_text = "<noinput>" if task["input"].lower() == "" else task["input"]
                
                examples_text += f"###\n"
                examples_text += f"Instruction: {instruction}\n"
                examples_text += f"Input: {input_text}\n"
                examples_text += f"Output: {task['output']}\n"
            
            examples_text += "\nNow generate 20 new, diverse task instructions following the same format:"
            messages.append({"role": "user", "content": examples_text})
            prompts.append(messages)
            
        return prompts

    async def generate_instructions_batch(self, messages_batch: List[List[Dict]]) -> List[Dict]:
        """异步批量生成指令"""
        async def generate_single(messages):
            try:
                response = await self.client.chat.completions.create(
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

        tasks = [generate_single(messages) for messages in messages_batch]
        results = await asyncio.gather(*tasks)
        return [item for sublist in results for item in sublist]

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
            
        item["instruction"] = item["instruction"].strip()
        item["input"] = "" if item["input"].lower() == "<noinput>" else item["input"].strip()
        item["output"] = item["output"].strip()
        
        if len(item["instruction"].split()) <= 3 or len(item["instruction"].split()) > 150:
            return False
            
        blacklist = [
            "image", "images", "graph", "graphs", "picture", "pictures",
            "file", "files", "map", "maps", "draw", "plot", "go to",
            "video", "audio", "music", "flowchart", "diagram"
        ]
        if any(word in item["instruction"].lower() for word in blacklist):
            return False
            
        if item["instruction"][0] in string.punctuation:
            return False
            
        if not item["instruction"][0].isascii():
            return False
            
        return True

    def check_similarity(self, new_instruction: str, threshold: float = 0.7) -> bool:
        """检查新指令与最近指令的相似度"""
        if not self.recent_instructions:
            return True
            
        for existing in self.recent_instructions:
            scores = self.scorer.score(existing, new_instruction)
            if scores['rougeL'].fmeasure > threshold:
                return False
        return True

    async def generate_dataset(self, 
                           seed_file: str,
                           num_instructions: int,
                           output_file: str) -> None:
        """异步生成完整数据集"""
        seed_tasks = self.load_seed_tasks(seed_file)
        print(f"加载了 {len(seed_tasks)} 个种子任务")
        
        dataset = []
        pbar = tqdm.tqdm(total=num_instructions)
        
        while len(dataset) < num_instructions:
            messages_batch = self.create_prompts_batch(seed_tasks, self.batch_size)
            new_instructions = await self.generate_instructions_batch(messages_batch)
            
            for item in new_instructions:
                if self.check_similarity(item["instruction"]):
                    dataset.append(item)
                    self.recent_instructions.append(item["instruction"])
                    pbar.update(1)
                    
                    if len(dataset) % 100 == 0:
                        self.save_dataset(dataset, output_file)
                        
                if len(dataset) >= num_instructions:
                    break
        
        self.save_dataset(dataset, output_file)
        print(f"已生成 {len(dataset)} 条指令并保存到 {output_file}")

    def save_dataset(self, dataset: List[Dict], filename: str) -> None:
        """保存数据集"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

async def main():
    model_name = "deepseek-chat"
    seed_file = "data/seed_tasks.jsonl"
    output_file = "data/alpaca_data.json"
    num_instructions = 2000
    
    generator = AlpacaDataGenerator(
        model_name=model_name,
        batch_size=500,  
        max_workers=256
    )
    
    await generator.generate_dataset(
        seed_file=seed_file,
        num_instructions=num_instructions,
        output_file=output_file
    )

if __name__ == "__main__":
    asyncio.run(main())