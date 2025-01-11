import json
import argparse
import random
import os
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="../data/alpaca_data.json")
    parser.add_argument("--output_dir", type=str, default="../data")
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    
    dataset = []
    with open(args.input_file, "r") as f:
        examples = json.load(f)
        for example in tqdm(examples):
            d = {}
            d['instruction'] = example['instruction'] + (("\nInput: " + example['input']) if example['input'] != "" else "")
            d['output'] = example['output']
            dataset.append(d)
    test_start = int(len(dataset) *  (1 - args.test_ratio))
    random.shuffle(dataset)
    train = dataset[:test_start]
    test = dataset[test_start:]
    with open(os.path.join(args.output_dir, "train.json"), "w") as f1, open(os.path.join(args.output_dir, "test.json"), "w") as f2:
        json.dump(train, f1, indent=2, sort_keys=True)
        json.dump(test, f2, indent=2, sort_keys=True)
    print(f"{len(train)} training examples.")
    print(f"{len(test)} test examples.")

if __name__ == "__main__":
    main()