# fin-rlhf

Generate dataset with ``python feedback/generate.py``
Annotate dataset with ``python feedback/annotate.py``

Run DPO with ``accelerate launch dpo.py --output_dir="dpo" --train_pct=0.01 --max_steps=5 --logging_steps=1 --eval_steps=5 --save_steps=5``