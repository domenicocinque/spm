#!/bin/bash

python src/train.py -m hparams_search=default_optuna experiment=proteins logger=wandb logger.wandb.group="proteins_optuna"
python src/train.py -m hparams_search=default_optuna experiment=dd logger=wandb logger.wandb.group="dd_optuna"
python src/train.py -m hparams_search=default_optuna experiment=msrc21 logger=wandb logger.wandb.group="msrc_optuna"
python src/train.py -m hparams_search=default_optuna experiment=nci1 logger=wandb logger.wandb.group="nci1_optuna"
python src/train.py -m hparams_search=default_optuna experiment=nci109 logger=wandb logger.wandb.group="nci109_optuna"

