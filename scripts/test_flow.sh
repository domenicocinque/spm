python src/train.py --multirun experiment=flow \
model.net.pooling_type=none \
seed=111,222,333,444,555  \
logger.wandb.group="nopool_flow_eval"

python src/train.py --multirun experiment=flow \
model.net.pooling_type=topk \
seed=111,222,333,444,555 \
logger.wandb.group="topk_flow_eval"

python src/train.py --multirun experiment=flow \
model.net.pooling_type=sagplus \
seed=111,222,333,444,555 \
logger.wandb.group="sagplus_flow_eval"

python src/train.py --multirun experiment=flow \
model.net.pooling_type=sag \
seed=111,222,333,444,555 \
logger.wandb.group="sag_flow_eval"


python src/train.py --multirun experiment=flow \
model.net.pooling_type=random  \
seed=111,222,333,444,555  \
logger.wandb.group="random_flow_eval"

python src/train.py --multirun experiment=flow \
model.net.pooling_type=max  \
seed=111,222,333,444,555  \
logger.wandb.group="max_flow_proteins"

python src/train.py --multirun experiment=flow \
model.net.pooling_type=sep_topk \
seed=111,222,333,444,555 \
logger.wandb.group="septopk_flow_eval"