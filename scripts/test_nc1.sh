python src/train.py --multirun experiment=nci1 \
model.net.pooling_type=none \
seed=111,222,333,444,555  \
logger.wandb.group="nopool_nci1_eval"

python src/train.py --multirun experiment=nci1 \
model.net.pooling_type=topk \
seed=111,222,333,444,555 \
logger.wandb.group="topk_nci1_eval"

python src/train.py --multirun experiment=nci1 \
model.net.pooling_type=sagplus \
seed=111,222,333,444,555 \
logger.wandb.group="sagplus_nci1_eval"

python src/train.py --multirun experiment=nci1 \
model.net.pooling_type=sag \
seed=111,222,333,444,555 \
logger.wandb.group="sag_nci1_eval"

python src/train.py --multirun experiment=nci1 \
model.net.pooling_type=septopk \
seed=111,222,333,444,555 \
logger.wandb.group="septopk_nci1_eval"

python src/train.py --multirun experiment=nci1 \
model.net.pooling_type=random  \
seed=111,222,333,444,555  \
logger.wandb.group="random_nci1_eval"

python src/train.py --multirun experiment=nci1 \
model.net.pooling_type=max  \
seed=111,222,333,444,555  \
logger.wandb.group="max_nci1_eval"


python src/train.py --multirun experiment=nci1 \
model.net.pooling_type=sep_topk \
seed=111,222,333,444,555 \
logger.wandb.group="septopk_nci1_eval"