import hydra
from omegaconf import DictConfig, OmegaConf
from config.struct_config import Exp_Config

@hydra.main(version_base=None,config_path='../config',config_name='config')
def run(cfg: DictConfig) -> None:
    cfg=OmegaConf.to_object(cfg)
    if isinstance(cfg,Exp_Config):
        print('Success')
    else:
        print('Failed\n')
        raise TypeError('configuration type not supported')
