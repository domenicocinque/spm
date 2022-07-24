python src/train.py --multirun experiment=ocean \
model.net.pooling_type=none \
seed=111,222,333,444,555  \
logger.wandb.group="nopool_ocean_eval"

python src/train.py --multirun experiment=ocean \
model.net.pooling_type=topk \
seed=111,222,333,444,555 \
logger.wandb.group="topk_ocean_eval"

python src/train.py --multirun experiment=ocean \
model.net.pooling_type=sagplus \
seed=111,222,333,444,555 \
logger.wandb.group="sagplus_ocean_eval"

python src/train.py --multirun experiment=ocean \
model.net.pooling_type=sag \
seed=111,222,333,444,555 \
logger.wandb.group="sag_ocean_eval"


python src/train.py --multirun experiment=ocean \
model.net.pooling_type=random  \
seed=111,222,333,444,555  \
logger.wandb.group="random_ocean_eval"

python src/train.py --multirun experiment=ocean \
model.net.pooling_type=max  \
seed=111,222,333,444,555  \
logger.wandb.group="max_ocean_proteins"

python src/train.py --multirun experiment=ocean \
model.net.pooling_type=sep_topk \
seed=111,222,333,444,555 \
logger.wandb.group="septopk_ocean_eval"