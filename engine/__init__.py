import hydra
from omegaconf import DictConfig, OmegaConf
from config.struct_config import Exp_Config

@hydra.main(version_base=None, config_path='../config', config_name='config')
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.to_object(cfg)
    global exp_used, dataset_used, exp, model_used, epochs, metrics, s_l_p, train_param, test_param, train_batch_size, test_batch_size, backup_chkpts
    exp = getattr(cfg.experiment,cfg.experiment_name)
    model = getattr(getattr(cfg.experiment,cfg.experiment_name),cfg.model_used)
    exp_used = cfg.experiment_name
    dataset_used = cfg.dataset_used
    model_used = cfg.model_used
    train_param = getattr(getattr(cfg.experiment,exp_used),model_used).train_param
    test_param = getattr(getattr(cfg.experiment,exp_used),model_used).test_param
    epochs = train_param.epochs
    metrics = train_param.metrics
    s_l_p = train_param.show_plot
    train_batch_size = model.train_param.batch_size
    if exp_used == 'super_resolution':
        test_batch_size = model.test_param.batch_size
    backup_chkpts = cfg.backup_checkpoints
main()