# LAYERS 


python src/plots.py plot_type='layers' experiment=ocean_plot_small \
model.net.pooling_type=sep_topk \
logger.wandb.group="layers_septopk_ocean"

python src/plots.py plot_type='layers' experiment=flow_plot_small \
model.net.pooling_type=max \
logger.wandb.group="layers_max_flow"

python src/plots.py plot_type='layers' experiment=flow_plot_small \
model.net.pooling_type=sep_topk \
logger.wandb.group="layers_septopk_flow"

# RATIOS

python src/plots.py plot_type='ratios' experiment=flow_plot_small \
model.net.pooling_type=sep_topk \
logger.wandb.group="ratios_septopk_flow"

python src/plots.py plot_type='ratios' experiment=ocean_plot_small \
model.net.pooling_type=max \
logger.wandb.group="ratios_max_ocean"

python src/plots.py plot_type='ratios' experiment=ocean_plot_small \
model.net.pooling_type=sep_topk \
logger.wandb.group="ratios_septopk_ocean"


