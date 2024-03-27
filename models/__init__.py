import hydra
from omegaconf import OmegaConf, DictConfig
from config.struct_config import Exp_Config

@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg: DictConfig):
    cfg=OmegaConf.to_object(cfg)
    global lr_shape, hr_shape, device, ddp, exp_name
    exp_name = cfg.experiment_name
    if exp_name == 'super_resolution':
        lr_shape = getattr(cfg.experiment,cfg.experiment_name).lr_shape
        hr_shape = getattr(cfg.experiment,cfg.experiment_name).hr_shape
    device = cfg.device
    ddp = cfg.ddp
main()