# fin-rlhf

Run SFT with: ``accelerate launch sft.py``

Merge SFT LORA weights with ``python merge.py --checkpoint_path="results/sft-finqa/checkpoint-550" --merged_path="results/sft-finqa/final_model``

Generate dataset with ``accelerate launch feedback/generate.py --model_name="results/sft-finqa/final_model" --tokenizer_name="mistralai/Mistral-7B-v0.1" --dataset_name="gbharti/finance-alpaca" --save_path="feedback/finance-alpaca-unlabeled.csv" --num_steps=100``

Annotate dataset with ``python feedback/annotate.py --unlabeled_path="feedback/finance-alpaca-unlabeled.csv" --labels_path="feedback/finance-alpaca-labels.csv"``

Merge generations with annotations and upload to huggingface with ``python feedback/merge_labels.py --hf_repo="danyoung/finance-feedback"``

Run DPO with ``accelerate launch dpo.py --output_dir="dpo" --train_pct=0.01 --max_steps=5 --logging_steps=1 --eval_steps=5 --save_steps=5``

Upload model with ``huggingface-cli upload danyoung/finance-qa results/sft-finqa/final_model``

Evaluate a model with ``accelerate launch evaluation/evaluate.py --model_name="results/sft-finqa/final_model" --tokenizer_name="mistralai/Mistral-7B-v0.1"``