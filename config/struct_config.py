from pydantic.dataclasses import dataclass
from dataclasses import field
from pydantic import model_validator
from typing import List
from hydra.core.config_store import ConfigStore

@dataclass
class SR_Dataset:
    path: List[str] = field(default_factory=list)
    download_data: bool = field(default=False)
    url: str = field(default_factory=str)
    train_test_split: List[float] = field(default_factory=list)

@dataclass
class DIFF_Dataset:
    path: str = field(default_factory=str)
    download_data: bool = field(default=False)
    url: str = field(default_factory=str)

@dataclass
class SRCNN_Model_Arch:
    encoder_channel_sequence: list = field(default_factory=lambda: [1,32,64,128,256])
    decoder_channel_sequence: list = field(default_factory=lambda: [256,128,64,32,16,1])

@dataclass
class SRCNN_Train_Param:
    weight_init: str = field(default='xavier_uniform')
    bias_init: str = field(default='zeros')
    optimizer: str = field(default='Adam')
    learning_rate: float = field(default_factory=float)
    betas: List[float] = field(default_factory=lambda: [0.9,0.99])
    weight_decay: float = field(default=0)
    batch_size: int = field(default_factory=int)
    show_plot: bool = field(default=False)
    metrics: List[str] = field(default_factory=lambda: ['mse','ssim','psnr'])
    epochs: int = field(default_factory=int)

    @classmethod
    def registers(cls):
        cls.init_register = ['xavier_uniform','xavier_normal',
                            'kaiming_uniform','kaiming_normal',
                            'uniform','normal',
                            'ones','zeros']
        
        cls.optim_register = ['Adam','SGD','RMSprop']

        cls.metrics_register = ['mse','ssim','psnr','accuracy','ROC','AUC']

    @model_validator(mode='after')
    @classmethod
    def specs_registry(cls,value):

        wi = value.weight_init
        bi = value.bias_init
        opt = value.optimizer
        met = value.metrics

        if wi not in cls.init_register:
            raise ValueError(f'No {wi} method registered')
        if bi not in cls.init_register:
            raise ValueError(f'No {bi} method registered')
        if opt not in cls.optim_register:
            raise ValueError(f'No {opt} algorithm registered')
        if not all(x in cls.metrics_register for x in met):
            raise ValueError(f'One or more metrics in {met} not registered')
        
        return value
        
@dataclass
class SRCNN_Test_Param:
    batch_size: int = field(default_factory=int)

@dataclass
class SRGAN_Model_Arch:
    discriminator_channel_sequence: list = field(default_factory=list)
    generator_residual_blocks: int = field(default_factory=int)
    residual_channel_number: int = field(default_factory=int)
    upsample_architecture: list = field(default_factory=list)

@dataclass
class SRGAN_Train_Param:
    gen_weight_init: str = field(default='xavier_uniform')
    gen_bias_init: str = field(default='zeros')
    gen_optimizer: str = field(default='Adam')
    gen_learning_rate: float = field(default_factory=float)
    gen_betas: List[float] = field(default_factory=lambda: [0.9,0.99])
    gen_weight_decay: float = field(default_factory=float)

    dis_weight_init: str = field(default='xavier_uniform')
    dis_bias_init: str = field(default='zeros')
    dis_optimizer: str = field(default='Adam')
    dis_learning_rate: float = field(default_factory=float)
    dis_betas: List[float] = field(default_factory=lambda: [0.9,0.99])
    dis_weight_decay: float = field(default_factory=float)

    batch_size: int = field(default_factory=int)
    show_plot: bool = field(default_factory=bool)
    metrics: List[str] = field(default_factory=lambda: ['mse','ssim','psnr'])
    epochs: int = field(default_factory=int)
    pretrain: bool = field(default=False)

    @classmethod
    def registers(cls):
        cls.init_register = ['xavier_uniform','xavier_normal',
                            'kaiming_uniform','kaiming_normal',
                            'uniform','normal',
                            'ones','zeros']
        
        cls.optim_register = ['Adam','SGD','RMSprop']

        cls.metrics_register = ['mse','ssim','psnr','accuracy','ROC','AUC']

    @model_validator(mode='after')
    @classmethod
    def specs_registry(cls,value):

        gwi = value.gen_weight_init
        dwi = value.dis_weight_init

        gbi = value.gen_bias_init
        dbi = value.dis_bias_init

        gopt = value.gen_optimizer
        dopt = value.dis_optimizer

        met = value.metrics

        if gwi not in cls.init_register:
            raise ValueError(f'No {gwi} method registered')
        if dwi not in cls.init_register:
            raise ValueError(f'No {dwi} method registered')
        
        if gbi not in cls.init_register:
            raise ValueError(f'No {gbi} method registered')
        if dbi not in cls.init_register:
            raise ValueError(f'No {dbi} method registered')
        
        if gopt not in cls.optim_register:
            raise ValueError(f'No {gopt} algorithm registered')
        if dopt not in cls.optim_register:
            raise ValueError(f'No {dopt} algorithm registered')
        
        if not all(x in cls.metrics_register for x in met):
            raise ValueError(f'No {met} metric registered')
        
        return value
    
@dataclass
class DIFF_UNET_Train_Param:
    weight_init: str = field(default='xavier_uniform')
    bias_init: str = field(default='zeros')
    optimizer: str = field(default='Adam')
    learning_rate: float = field(default_factory=float)
    betas: List[float] = field(default_factory=lambda: [0.9,0.99])
    weight_decay: float = field(default=0)
    batch_size: int = field(default_factory=int)
    show_plot: bool = field(default=False)
    metrics: List[str] = field(default_factory=lambda: ['fid'])
    epochs: int = field(default_factory=int)

    @classmethod
    def registers(cls):
        cls.init_register = ['xavier_uniform','xavier_normal',
                            'kaiming_uniform','kaiming_normal',
                            'uniform','normal',
                            'ones','zeros']
        
        cls.optim_register = ['Adam','SGD','RMSprop']

        cls.metrics_register = ['mse']

    @classmethod
    @model_validator(mode="after")
    def specs_registry(cls,value):

        wi = value.weight_init
        bi = value.bias_init
        opt = value.optimizer
        met = value.metrics

        if wi not in cls.init_register:
            raise ValueError(f'No {wi} method registered')
        if bi not in cls.init_register:
            raise ValueError(f'No {bi} method registered')
        if opt not in cls.optim_register:
            raise ValueError(f'No {opt} algorithm registered')
        if not all(x in cls.metrics_register for x in met):
            raise ValueError(f'One or more metrics in {met} not registered')
        
        return value

@dataclass
class SRGAN_Test_Param:
    batch_size: int = field(default_factory=int)

@dataclass
class DIFF_UNET_Test_Param:
    samples: int = field(default_factory=int)
    sampling_technique: str = field(default_factory=str)

    @classmethod
    def registers(cls):
        cls.sampling_register = ['ddpm', 'ddim']

    @classmethod
    @model_validator(mode='after')
    def specs_registry(cls, value):
        samp = value.sampling_technique
        if samp not in cls.sampling_register:
            raise ValueError(f'No {samp} sampling technique registered')

@dataclass
class SRCNN:
    model_arch: SRCNN_Model_Arch = field(default_factory=SRCNN_Model_Arch)
    train_param: SRCNN_Train_Param = field(default_factory=SRCNN_Train_Param)
    test_param: SRCNN_Test_Param = field(default_factory=SRCNN_Test_Param)

@dataclass
class SRGAN:
    model_arch: SRGAN_Model_Arch = field(default_factory=SRGAN_Model_Arch)
    train_param: SRGAN_Train_Param = field(default_factory=SRGAN_Train_Param)
    test_param: SRGAN_Test_Param = field(default_factory=SRGAN_Test_Param)

@dataclass
class DIFF_UNET:
    train_param: DIFF_UNET_Train_Param = field(default_factory=DIFF_UNET_Train_Param)
    test_param: DIFF_UNET_Test_Param = field(default_factory=DIFF_UNET_Test_Param)

@dataclass
class SR:
    lr_shape: list = field(default_factory=list)
    hr_shape: list = field(default_factory=list)
    dataset: SR_Dataset = field(default_factory=SR_Dataset)
    srcnn: SRCNN = field(default_factory=SRCNN)
    srgan: SRGAN = field(default_factory=SRGAN)

@dataclass
class DIFF:
    shape: list = field(default_factory=list)
    timesteps: int = field(default_factory=int)
    noising_schedule: str = field(default_factory=str)
    dataset: DIFF_Dataset = field(default_factory=DIFF_Dataset)
    diff_unet: DIFF_UNET = field(default_factory=DIFF_UNET)

    @classmethod
    def registers(cls):
        cls.noising_schedule_register = ['linear', 'cosine']

    @classmethod
    @model_validator(mode="after")
    def diff_registry(cls, value):
        ns = value.noising_schedule
        if ns not in cls.noising_schedule_register:
            raise ValueError(f'No {ns} schedule is registered')

@dataclass
class Experiment:
    super_resolution: SR = field(default_factory=SR)
    diffusion: DIFF = field(default_factory=DIFF)

@dataclass
class Exp_Config:
    experiment_name: str = field(default_factory=str)
    model_used: str = field(default_factory=str)
    dataset_used: str = field(default='dataset')
    device: str = field(default='cpu')
    ddp: bool = field(default=False)
    backup_checkpoints: int = field(default_factory=int)
    experiment: Experiment = field(default_factory=Experiment)

    @classmethod
    def registers(cls):
        cls.experiment_register = ['super_resolution','diffusion']
        cls.model_register = ['srcnn','srgan','diff_unet']
        cls.device_register = ['cpu','gpu']
        SRCNN_Train_Param.registers()
        SRGAN_Train_Param.registers()
        DIFF_UNET_Train_Param.registers()
        DIFF_UNET_Test_Param.registers()

    @model_validator(mode='after')
    @classmethod
    def exp_registry(cls,value):

        exp = value.experiment_name
        mod = value.model_used
        dev = value.device

        if exp not in cls.experiment_register:
            raise ValueError(f'No {exp} experiment is registered')
        if mod not in cls.model_register:
            raise ValueError(f'No {mod} model registered')
        if dev not in cls.device_register:
            raise ValueError(f'No {dev} device registered')

Exp_Config.registers()
cs = ConfigStore.instance()
cs.store(name='struct_config',node=Exp_Config)


