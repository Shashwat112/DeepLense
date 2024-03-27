import hydra
from omegaconf import DictConfig, OmegaConf
from config.struct_config import Exp_Config

@hydra.main(version_base=None, config_path='../config', config_name='config')
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.to_object(cfg)
    global train_param, discriminator_channel_sequence, generator_residual_blocks, residual_channel_number, upsample_architecture
    mod = getattr(cfg.experiment,cfg.experiment_name).srgan
    train_param = mod.train_param
    discriminator_channel_sequence = mod.model_arch.discriminator_channel_sequence
    generator_residual_blocks = mod.model_arch.generator_residual_blocks
    residual_channel_number = mod.model_arch.residual_channel_number
    upsample_architecture = mod.model_arch.upsample_architecture
main()
