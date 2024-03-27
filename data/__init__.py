import hydra
from omegaconf import OmegaConf,DictConfig
from config.struct_config import Exp_Config
from pathlib import Path

@hydra.main(version_base=None,config_path='config',config_name='config')
def main(cfg: DictConfig):
    cfg = OmegaConf.to_object(cfg)
    global exp_name, path, download_data, url, train_test_split, train_batch_size, test_batch_size, ddp
    exp_name = cfg.experiment_name
    dataset = getattr(getattr(cfg.experiment,cfg.experiment_name),cfg.dataset_used)
    model = getattr(getattr(cfg.experiment,cfg.experiment_name),cfg.model_used)
    path = dataset.path
    download_data = dataset.download_data
    url = dataset.url
    train_batch_size = model.train_param.batch_size
    if exp_name == 'super_resolution':
        train_test_split = dataset.train_test_split
        test_batch_size = model.test_param.batch_size
    ddp = cfg.ddp
main()