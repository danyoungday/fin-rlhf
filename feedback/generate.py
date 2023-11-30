import argparse
import os
import pandas as pd
from tqdm import tqdm
import csv

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

import torch
from torch.utils.data import DataLoader

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        help="the model to generate responses from")
    parser.add_argument("--dataset_name",
                        default="gbharti/finance-alpaca",
                        help="the dataset to generate responses to")
    parser.add_argument("--cache_dir",
                        default="cache_dir",
                        help="where to cache the huggingface objects")
    parser.add_argument("--save_path",
                        default="feedback/unlabeled.csv",
                        help="path to save the unlabeled data to")
    parser.add_argument("--batch_size",
                        type=int,
                        default=4,
                        help="number of prompts to generate responses for at once")
    parser.add_argument("--num_batches",
                        type=int,
                        default=-1,
                        help="number of batches to save. -1 means do not stop.")
    parser.add_argument("--write_steps",
                        type=int,
                        default=5,
                        help="number of batches to iterate through before saving as a chunk")
    args = parser.parse_args()
    return args


def generate_response_pair(model, tokenizer, batch):
    """
    Generates a pair of responses given a batch of prompts, a model, and a tokenizer.
    Input should be shape N list of strings
    Output is 3 x N lists of strings
    """
    tokens = tokenizer(batch, padding=True, return_tensors="pt").to(model.device)
    prompt_len = tokens.input_ids.shape[1]
    out_a = model.generate(**tokens, max_new_tokens=16, do_sample=True, pad_token_id=50256)
    out_b = model.generate(**tokens, max_new_tokens=16, do_sample=True, pad_token_id=50256)
    prompts = tokenizer.batch_decode(out_a[:,:prompt_len], skip_special_tokens=True) 
    responses_a = tokenizer.batch_decode(out_a[:,prompt_len:], skip_special_tokens=True)
    responses_b = tokenizer.batch_decode(out_b[:,prompt_len:], skip_special_tokens=True)
    return prompts, responses_a, responses_b


if __name__=="__main__":
    args = create_argparser()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left", cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token

    # If the file doesn't exist, create the header
    if not os.path.exists(args.save_path):
        with(open(args.save_path, "w")) as f:
            f.write("prompt,response_a,response_b\n")

    # Read to the correct number in the dataset
    n = sum(1 for _ in open(args.save_path)) - 1
    ds = load_dataset(args.dataset_name, cache_dir=args.cache_dir)
    train = ds["train"][n:]
    
    with open(args.save_path, "a", newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        write_batch = []
        dl = DataLoader(train["instruction"], batch_size=args.batch_size, shuffle=False)
        for step, batch in tqdm(enumerate(dl)):
            if step == args.num_batches:
                break

            # Generate responses for batch
            prompts, responses_a, responses_b = generate_response_pair(model, tokenizer, batch)

            # Batch out outputs so we minimize the amount of file writes.
            # Lines are wrapped in quotes so the commas can go in the csv.
            write_batch += list(zip(prompts, responses_a, responses_b))
            if (step + 1) % args.write_steps == 0:
                writer.writerows(write_batch)
                write_batch.clear()

        writer.writerows(write_batch)