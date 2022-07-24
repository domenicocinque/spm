python src/train.py --multirun experiment=msrc21_gnn \
model.net.pooling_type=graph_topk  \
seed=111,222,333,444,555 \
logger.wandb.group="gnn_topk_msrc21_eval"


python src/train.py --multirun experiment=msrc21_gnn \
model.net.pooling_type=graph_sag \
seed=111,222,333,444,555 \
logger.wandb.group="gnn_sag_msrc21_eval"
