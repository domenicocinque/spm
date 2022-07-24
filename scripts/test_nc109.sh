python src/train.py --multirun experiment=nci109 \
model.net.pooling_type=none \
seed=111,222,333,444,555  \
logger.wandb.group="nopool_nci109_eval"

python src/train.py --multirun experiment=nci109 \
model.net.pooling_type=topk \
seed=111,222,333,444,555 \
logger.wandb.group="topk_nci109_eval"

python src/train.py --multirun experiment=nci109 \
model.net.pooling_type=sagplus \
seed=111,222,333,444,555 \
logger.wandb.group="sagplus_nci109_eval"

python src/train.py --multirun experiment=nci109 \
model.net.pooling_type=sag \
seed=111,222,333,444,555 \
logger.wandb.group="sag_nci109_eval"

python src/train.py --multirun experiment=nci109 \
model.net.pooling_type=septopk \
seed=111,222,333,444,555 \
logger.wandb.group="septopk_nci109_eval"

python src/train.py --multirun experiment=nci109 \
model.net.pooling_type=random  \
seed=111,222,333,444,555  \
logger.wandb.group="random_nci109_eval"

python src/train.py --multirun experiment=nci109 \
model.net.pooling_type=max  \
seed=111,222,333,444,555  \
logger.wandb.group="max_nci109_eval"

python src/train.py --multirun experiment=nci109 \
model.net.pooling_type=sep_topk \
seed=111,222,333,444,555 \
logger.wandb.group="septopk_nci109_eval"