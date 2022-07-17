python src/train.py --multirun datamodule=proteins \
experiment=nopool seed=111,222,333,444,555 \
logger=wandb logger.wandb.group="nopool_proteins_eval" \
trainer=gpu

python src/train.py --multirun datamodule=proteins \
experiment=topk seed=111,222,333,444,555 \
logger=wandb logger.wandb.group="topk_proteins_eval" \
trainer=gpu

python src/train.py --multirun datamodule=proteins \
experiment=sag seed=111,222,333,444,555 \
logger=wandb logger.wandb.group="sag_proteins_eval" \
trainer=gpu

python src/train.py --multirun datamodule=proteins \
experiment=sagplus seed=111,222,333,444,555 \
logger=wandb logger.wandb.group="sagplus_proteins_eval" \
trainer=gpu

python src/train.py --multirun datamodule=proteins \
experiment=septopk seed=111,222,333,444,555 \
logger=wandb logger.wandb.group="septopk_proteins_eval" \
trainer=gpu
