import hydra
from omegaconf import DictConfig, OmegaConf
from config.struct_config import Exp_Config

@hydra.main(version_base=None, config_path='../config', config_name='config')
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.to_object(cfg)
    global train_param, timesteps, noise_sched, test_param
    timesteps = getattr(cfg.experiment,cfg.experiment_name).timesteps
    noise_sched = getattr(cfg.experiment,cfg.experiment_name).noising_schedule
    mod = getattr(cfg.experiment,cfg.experiment_name).diff_unet
    train_param = mod.train_param
    test_param = mod.test_param
main()
