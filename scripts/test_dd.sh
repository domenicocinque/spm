python src/train.py --multirun experiment=dd \
model.net.pooling_type=topk \
seed=111,222,333,444,555  \
trainer.max_epochs=50 callbacks.early_stopping.patience=10 \
logger.wandb.group="topk_dd_eval_3" 

python src/train.py --multirun experiment=dd \
model.net.pooling_type=sagplus \
seed=111,222,333,444,555 \
trainer.max_epochs=50 callbacks.early_stopping.patience=10 \
logger.wandb.group="sagplus_dd_eval_3" 

python src/train.py --multirun experiment=dd \
model.net.pooling_type=sag \
seed=111,222,333,444,555  \
trainer.max_epochs=50 callbacks.early_stopping.patience=10 \
logger.wandb.group="sag_dd_eval_3" 

python src/train.py --multirun experiment=dd \
model.net.pooling_type=random  \
seed=111,222,333,444,555 \
trainer.max_epochs=50 callbacks.early_stopping.patience=10 \
logger.wandb.group="random_dd_eval_3" 

python src/train.py --multirun experiment=dd \
model.net.pooling_type=max  \
seed=111,222,333,444,555 \
trainer.max_epochs=50 callbacks.early_stopping.patience=10 \
logger.wandb.group="max_dd_eval_3"

python src/train.py --multirun experiment=dd \
model.net.pooling_type=sep_topk \
seed=111,222,333,444,555  \
trainer.max_epochs=50 callbacks.early_stopping.patience=10 \
logger.wandb.group="septopk_dd_eval_3" 

