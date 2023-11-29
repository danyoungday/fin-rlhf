import pandas as pd
import os

if __name__ == "__main__":
    unlabeled_path = "feedback/unlabeled.csv"
    labels_path = "feedback/labels.csv"

    if not os.path.exists(labels_path):
        with open(labels_path, "w") as f:
            f.write("label\n")
    
    df = pd.read_csv(unlabeled_path)
    n = sum(1 for _ in open(labels_path)) - 1
    with open(labels_path, "a") as f:
        for i in range(n, len(df)):
            row = df.iloc[i]
            print(f"Prompt: {repr(row['prompt'])}")
            print(f"Response A: {repr(row['response_a'])}")
            print(f"Response B: {repr(row['response_b'])}")
            label = input("1 for prompt A, 2 for prompt B")
            f.write(f"{label-1}\n")
