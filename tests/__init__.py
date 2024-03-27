import hydra
from config.struct_config import Exp_Config
from engine import exp_used, model_used

@hydra.main(version_base=None, config_path='../config', config_name='test_config')
def main(cfg) -> None:
    global model_register, model_used, experiment_used, desired_dataloader_lr_shape, desired_dataloader_hr_shape, desired_model_output_shape, desired_discriminator_output_shape, desired_generator_output_shape, desired_dataloader_shape
    model_register = Exp_Config.model_register
    model_used = model_used
    experiment_used = exp_used
    if experiment_used == 'super_resolution':
        desired_dataloader_lr_shape = cfg['experiment'][exp_used][model_used]['desired_dataloader_lr_shape']
        desired_dataloader_hr_shape = cfg['experiment'][exp_used][model_used]['desired_dataloader_hr_shape']
        if model_used == 'srcnn':
            desired_model_output_shape = cfg['experiment'][exp_used][model_used]['desired_model_output_shape']
        if model_used == 'srgan':
            desired_discriminator_output_shape = cfg['experiment'][exp_used][model_used]['desired_discriminator_output_shape']
            desired_generator_output_shape = cfg['experiment'][exp_used][model_used]['desired_generator_output_shape']
    if experiment_used == 'diffusion':
        desired_dataloader_shape = cfg['experiment'][exp_used][model_used]['desired_dataloader_shape']
        desired_model_output_shape = cfg['experiment'][exp_used][model_used]['desired_model_output_shape']

main()