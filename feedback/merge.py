import pandas as pd
from datasets import load_dataset, Dataset
import argparse

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlabeled_path",
                        default="feedback/unlabeled.csv",
                        help="path to unlabeled prompt/abresponse data")
    parser.add_argument("--labels_path",
                        default="feedback/labels.csv",
                        help="path to the labels for the unlabeled data")
    parser.add_argument("--hf_repo",
                        default="danyoung/finance-feedback",
                        help="huggingface repo to save data to")
    parser.add_argument("--cache_dir",
                        default="cache_dir",
                        help="where to cache huggingface files")
    return parser.parse_args()

if __name__ == "__main__":
    args = create_argparser()
    labels = pd.read_csv(args.labels_path)
    df = pd.read_csv(args.unlabeled_path).iloc[:len(labels)]
    df["label"] = labels
    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub(args.hf_repo)

    validate = load_dataset(args.hf_repo, cache_dir=args.cache_dir)
    assert len(validate["train"]) == len(dataset), "Warning: loaded dataset length not same as constructed one."