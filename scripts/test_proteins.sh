python src/train.py --multirun experiment=proteins \
model.net.pooling_type=none \
seed=111,222,333,444,555  \
logger.wandb.group="nopool_proteins_eval_2"

python src/train.py --multirun experiment=proteins \
model.net.pooling_type=topk \
seed=111,222,333,444,555 \
logger.wandb.group="topk_proteins_eval_2"

python src/train.py --multirun experiment=proteins \
model.net.pooling_type=sagplus \
seed=111,222,333,444,555 \
logger.wandb.group="sagplus_proteins_eval_2"

python src/train.py --multirun experiment=proteins \
model.net.pooling_type=sag \
seed=111,222,333,444,555 \
logger.wandb.group="sag_proteins_eval_2"

python src/train.py --multirun experiment=proteins \
model.net.pooling_type=septopk \
seed=111,222,333,444,555 \
logger.wandb.group="septopk_proteins_eval_2"

python src/train.py --multirun experiment=proteins \
model.net.pooling_type=random  \
seed=111,222,333,444,555  \
logger.wandb.group="random_proteins_eval_2"

python src/train.py --multirun experiment=proteins \
model.net.pooling_type=max  \
seed=111,222,333,444,555  \
logger.wandb.group="max_proteins_eval_2"


python src/train.py --multirun experiment=proteins \
model.net.pooling_type=sep_topk \
seed=111,222,333,444,555 \
logger.wandb.group="septopk_proteins_eval_2"