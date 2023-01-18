python src/train.py --multirun experiment=proteins_gnn \
model.net.pooling_type=graph_topk \
seed=111,222,333,444,555  \
trainer.max_epochs=50 callbacks.early_stopping.patience=20 \
logger.wandb.group="gnn_topk_proteins_eval" \
logger.wandb.project="spm_check"

python src/train.py --multirun experiment=proteins_gnn \
model.net.pooling_type=graph_sag \
seed=111,222,333,444,555  \
trainer.max_epochs=50 callbacks.early_stopping.patience=20 \
logger.wandb.group="gnn_sag_proteins_eval" \
logger.wandb.project="spm_check"