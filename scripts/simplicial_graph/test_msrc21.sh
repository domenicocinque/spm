
python src/train.py --multirun experiment=msrc21 \
model.net.pooling_type=none \
seed=111,222,333,444,555  \
trainer.max_epochs=50 callbacks.early_stopping.patience=10 \
logger.wandb.group="nopool_msrc21_eval" \
logger.wandb.project="spm_check"

python src/train.py --multirun experiment=msrc21 \
model.net.pooling_type=max \
seed=111,222,333,444,555  \
trainer.max_epochs=50 callbacks.early_stopping.patience=10 \
logger.wandb.group="max_msrc21_eval" \
logger.wandb.project="spm_check"

python src/train.py --multirun experiment=msrc21 \
model.net.pooling_type=sag \
seed=111,222,333,444,555  \
trainer.max_epochs=50 callbacks.early_stopping.patience=10 \
logger.wandb.group="sag_msrc21_eval" \
logger.wandb.project="spm_check"

python src/train.py --multirun experiment=msrc21 \
model.net.pooling_type=topk \
seed=111,222,333,444,555  \
trainer.max_epochs=50 callbacks.early_stopping.patience=10 \
logger.wandb.group="topk_msrc21_eval" \
logger.wandb.project="spm_check"

python src/train.py --multirun experiment=msrc21 \
model.net.pooling_type=random \
seed=111,222,333,444,555  \
trainer.max_epochs=50 callbacks.early_stopping.patience=10 \
logger.wandb.group="random_msrc21_eval" \
logger.wandb.project="spm_check"

python src/train.py --multirun experiment=msrc21 \
model.net.pooling_type=sep_topk \
seed=111,222,333,444,555  \
trainer.max_epochs=50 callbacks.early_stopping.patience=10 \
logger.wandb.group="septopk_msrc21_eval" \
logger.wandb.project="spm_check"
