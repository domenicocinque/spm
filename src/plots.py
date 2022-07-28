import numpy as np
import hydra
import pyrootutils
from omegaconf import DictConfig

def save_array(array, filename):
    with open(filename, 'wb') as f:
        np.save(f, array)

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)

@hydra.main(version_base="1.2", config_path=root / "configs", config_name="plots.yaml")
def main(config: DictConfig):
    from src.tasks.train_task import train
    config_original = config.copy()

    results = []
    if config.plot_type == 'ratios':
        x_axis = [0.2, 0.4, 0.6, 0.8]
    elif config.plot_type == 'layers':
        x_axis = [2, 4, 6, 8, 10]
    else:
        raise ValueError('Unknown plot type: {}'.format(config.plot_type))

    seeds = [111, 222, 333, 444, 555]
    for x in x_axis:
        x_acc = []
        for seed in seeds:
            config = config_original.copy()
            config.seed = seed

            if config.plot_type == 'ratios':
                config.model.net.pooling_ratio = x
            elif config.plot_type == 'layers':
                config.model.net.hidden_layers = x

            metric_dict, _ = train(config)
            metric_value = metric_dict['test/acc']
            x_acc.append(metric_value)
        results.append(x_acc)
    save_array(results, f'{config.plot_type}_{config.model.net.pooling_type}_{config.name}.npy')
    return results


if __name__ == "__main__":
    main()