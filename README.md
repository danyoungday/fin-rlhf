# fin-rlhf

Generate dataset with ``python feedback/generate.py --model_name="mistralai/Mistral-7B-v0.1" --dataset_name="gbharti/finance-alpaca" --save_path="feedback/finance-alpaca-unlabeled.csv" --num_steps=5``

Annotate dataset with ``python feedback/annotate.py``

Merge generations with annotations and upload to huggingface with ``python feedback/merge.py --hf_repo="danyoung/finance-feedback"``

Run DPO with ``accelerate launch dpo.py --output_dir="dpo" --train_pct=0.01 --max_steps=5 --logging_steps=1 --eval_steps=5 --save_steps=5``