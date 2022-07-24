python src/train.py --multirun experiment=msrc21 \
model.net.pooling_type=none \
seed=111,222,333,444,555  \
logger.wandb.group="nopool_msrc21_eval"

python src/train.py --multirun experiment=msrc21 \
model.net.pooling_type=topk \
seed=111,222,333,444,555 \
logger.wandb.group="topk_msrc21_eval"

python src/train.py --multirun experiment=msrc21 \
model.net.pooling_type=sagplus \
seed=111,222,333,444,555 \
logger.wandb.group="sagplus_msrc21_eval"

python src/train.py --multirun experiment=msrc21 \
model.net.pooling_type=sag \
seed=111,222,333,444,555 \
logger.wandb.group="sag_msrc21_eval"

python src/train.py --multirun experiment=msrc21 \
model.net.pooling_type=septopk \
seed=111,222,333,444,555 \
logger.wandb.group="septopk_msrc21_eval"

python src/train.py --multirun experiment=msrc21 \
model.net.pooling_type=random  \
seed=111,222,333,444,555  \
logger.wandb.group="random_msrc21_eval"

python src/train.py --multirun experiment=msrc21 \
model.net.pooling_type=max  \
seed=111,222,333,444,555  \
logger.wandb.group="max_msrc21_eval"

python src/train.py --multirun experiment=msrc21 \
model.net.pooling_type=sep_topk \
seed=111,222,333,444,555 \
logger.wandb.group="septopk_msrc21_eval"