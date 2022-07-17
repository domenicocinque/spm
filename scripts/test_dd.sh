python src/train.py --multirun datamodule=dd \
experiment=nopool seed=111,222,333,444,555 \
logger=wandb logger.wandb.group="nopool_proteins_eval" \
trainer.gpus=1 

python src/train.py --multirun datamodule=dd \
experiment=topk seed=111,222,333,444,555 \
logger=wandb logger.wandb.group="topk_proteins_eval" \
trainer.gpus=1 

python src/train.py --multirun datamodule=dd \
experiment=sag seed=111,222,333,444,555 \
logger=wandb logger.wandb.group="sag_proteins_eval" \
trainer.gpus=1 

python src/train.py --multirun datamodule=dd \
experiment=sagplus seed=111,222,333,444,555 \
logger=wandb logger.wandb.group="sagplus_proteins_eval" \
trainer.gpus=1 

python src/train.py --multirun datamodule=dd \
experiment=septopk seed=111,222,333,444,555 \
logger=wandb logger.wandb.group="septopk_proteins_eval" \
trainer.gpus=1 