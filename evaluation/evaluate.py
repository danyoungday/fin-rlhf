from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import Accelerator
from datasets import load_dataset
import torch
from peft import LoraConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os

def create_dataset():
    dataset = load_dataset("gbharti/finance-alpaca", cache_dir="cache_dir", split="train[:24%]")
    # test_idx = range(int(len(dataset) * 0.8), len(dataset))
    test_idx = range(int(len(dataset) * 0.8), int(len(dataset) * 0.8) + 20)
    dataset = dataset.select(test_idx)
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.map(
        lambda example: {"text": 
                        f"<s>[INST] {example['instruction']} [/INST] {example['output']}<\s>"},
        num_proc=4)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        help="the model to evaluate")
    parser.add_argument("--tokenizer_name",
                        default=None,
                        help="the tokenizer's name")
    parser.add_argument("--save_path",
                        default="evaluation/results.csv",
                        help="where to save the results")
    parser.add_argument("--batch_size",
                        default=4,
                        type=int,
                        help="batch size")
    args = parser.parse_args()
    
    # Create results file if it doesn't already exist
    if not os.path.exists(args.save_path):
        with open(args.save_path, "w") as f:
            f.write("model_name,loss\n")

    # Model setup
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name, 
                                              cache_dir="cache_dir")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        use_nested_quant = True,
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_name, 
                                                        quantization_config=bnb_config,
                                                        device_map={"": Accelerator().local_process_index},
                                                        cache_dir="cache_dir")
    model.eval()

    ds = create_dataset()
    dl = DataLoader(ds["text"], batch_size=args.batch_size, shuffle=False)
    losses = []
    for batch in tqdm(dl):
        tokens = tokenizer(batch, padding=True, return_tensors="pt")
        mask = tokens["attention_mask"]
        out = model(**tokens)

    with open(args.save_path, "a") as f:
        f.write(f"{args.model_name},{sum(losses) / len(ds)}\n")
            

