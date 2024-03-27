import hydra
from omegaconf import DictConfig, OmegaConf
from config.struct_config import Exp_Config

@hydra.main(version_base=None, config_path='../config', config_name='config')
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.to_object(cfg)
    global train_param, encoder_channel_sequence, decoder_channel_sequence
    mod = getattr(cfg.experiment,cfg.experiment_name).srcnn
    train_param = mod.train_param
    encoder_channel_sequence = mod.model_arch.encoder_channel_sequence
    decoder_channel_sequence = mod.model_arch.decoder_channel_sequence
main()
