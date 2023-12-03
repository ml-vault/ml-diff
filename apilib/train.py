sd_scripts_path = "/workspace/difflex"
import os
from time import sleep
from typing import Literal, Optional
from abc import ABCMeta, abstractmethod
from apilib.util import run_cli
import subprocess
from apilib.util.env import SKIP_PROC, SDXL


alpha = "abcdefghijklmnopqrstuvwxyz"

def done():
    print("done!")

def get_name(name:str):
    filename, file_extension = os.path.splitext(name)
    return filename

def get_ext(name:str):
    filename, file_extension = os.path.splitext(name)
    return file_extension

def is_ckpt(name:str):
    ext = get_ext(name)
    return ext == ".safetensors" or ext == ".ckpt"

def add_line(file ,key: str, val:str) -> str:
    return file.write(f"{key}: {val}\n")

def trim_map(x: str) -> list[str]:
    return list(map(lambda x: x.strip(), x.split(",")))

class OutputConfig:
    base_path:str
    model_name:str
    save_every_n_epochs:int
    save_model_as:str

    def __init__(self, base_path:str, model_name:str, save_every_n_epochs:int, save_model_as: Literal["safetensors", "ckpt"] = "safetensors"):
        self.base_path = base_path
        self.model_name = model_name
        self.save_every_n_epochs = save_every_n_epochs
        self.save_model_as = save_model_as
        pass

    @property
    def out_dir(self):
        return f"{self.base_path}/{self.model_name}"

    def getArgs(self):
        os.makedirs(self.out_dir, exist_ok=True)
        return f"--output_dir {self.out_dir} \
        --output_name {self.model_name} \
        --save_every_n_epochs {self.save_every_n_epochs} \
        --save_model_as {self.save_model_as}"

class OptimizerConfig(metaclass=ABCMeta):
    optimizer_type:str
    def __init__(self, optimizer_type:Literal["AdaFactor","AdamW8bit" ]) -> None:
        self.optimizer_type = optimizer_type

    @abstractmethod
    def getArgs() -> str:
        pass

class AdamW8bitConfig(OptimizerConfig):
    def __init__(self):
        super().__init__("AdamW8bit")

    def getArgs(self) -> str:
        return f"--optimizer_type {self.optimizer_type}"


class AdaFactorConfig(OptimizerConfig):
    def __init__(self):
        super().__init__("AdaFactor")

    def getArgs(self) -> str:
        return f'--optimizer_type {self.optimizer_type} \
        --optimizer_args "relative_step=True" "scale_parameter=True" "warmup_init=False'

class TrainConfig:
    pretrained_model_name_or_path:str
    max_train_epochs:int
    train_batch_size:int
    network_dim:int
    network_alpha:int
    prior_loss_weight:float
    learning_rate:float
    mixed_precision:str
    network_module:str
    max_data_loader_n_workers:int
    config_file_path:str
    continue_from:Optional[str]
    def __init__(self, config_file_path:str, pretrained_model_name_or_path:str, max_train_epochs:int, train_batch_size:int, learning_rate:float, network_dim:int, network_alpha:int, 
                   mixed_precision: Literal["no", "fp16", "bf16"] = "bf16",
                   network_module:Literal["networks.lora", "lycoris.kohya"] = "networks.lora",
                   continue_from:Optional[str]=None,
                   max_data_loader_n_workers:int =3000,
                   prior_loss_weight=1.0) -> None:
        self.config_file_path = config_file_path
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.max_train_epochs = max_train_epochs
        self.train_batch_size = train_batch_size
        self.network_alpha = network_alpha
        self.network_dim = network_dim
        self.prior_loss_weight = prior_loss_weight
        self.learning_rate = learning_rate
        self.mixed_precision = mixed_precision
        self.network_module = network_module
        self.max_data_loader_n_workers = max_data_loader_n_workers
        self.continue_from = continue_from
        pass

    def getArgs(self) -> str:
        network_weight = f"--network_weight {self.continue_from} " if self.continue_from else ""
        dynamic = f"{network_weight}"
        return f"--dataset_config {self.config_file_path} \
        --pretrained_model_name_or_path {self.pretrained_model_name_or_path} \
        --max_train_epochs {self.max_train_epochs} \
        --train_batch_size {self.train_batch_size} \
        --network_dim {self.network_dim} --network_alpha {self.network_alpha} \
        --prior_loss_weight {self.prior_loss_weight} \
        --learning_rate {self.learning_rate} \
        --mixed_precision {self.mixed_precision} \
        --network_module {self.network_module} \
        --max_data_loader_n_workers {self.max_data_loader_n_workers} \
        {dynamic}"

class SampleConfig:
    sampler: str
    sample_every_n_epochs:int
    prompt_path:str
    def __init__(self,
               sampler: Literal['ddim', 'pndm', 'lms', 'euler', 'euler_a', 'heun', 'dpm_2', 'dpm_2_a', 'dpmsolver', 'dpmsolver++', 'dpmsingle', 'k_lms', 'k_euler', 'k_euler_a', 'k_dpm_2', 'k_dpm_2_a'],
               sample_every_n_epochs:int,
               prompt_path:str
               ) -> None:
        self.sampler = sampler
        self.sample_every_n_epochs = sample_every_n_epochs
        self.prompt_path = prompt_path
        pass
    def getArgs(self) -> str:
        return f"--sample_every_n_epochs {self.sample_every_n_epochs} --sample_prompts {self.prompt_path} --sample_sampler {self.sampler}"

def gen_train_lora_args(train_config:TrainConfig, output_config:OutputConfig, optimizer_config:OptimizerConfig, sample_config:Optional[SampleConfig] = None):
   
    basic_args = "--cache_latents --gradient_checkpointing --xformers"
    train_args = train_config.getArgs()
    output_args = output_config.getArgs()
    opt_args = optimizer_config.getArgs()
    sample_args = sample_config.getArgs() if sample_config else ""
    args = f"{basic_args} {train_args} {output_args} {opt_args} {sample_args}"
    return args

def train_lora(base_path:str,
                  config_file_path:str,
                  model_name:str,
                  save_every_n_epochs:int,
                  max_train_epochs:int,
                  train_batch_size:int,
                  learning_rate:float,
                  network_dim:int,
                  network_alpha:int
                 ):
    output_config = OutputConfig(
        base_path=base_path,
        model_name=model_name,
        save_every_n_epochs=save_every_n_epochs
    )
    train_config = TrainConfig(
        config_file_path=config_file_path,
        pretrained_model_name_or_path=SDXL,
        max_train_epochs=max_train_epochs,
        train_batch_size=train_batch_size,
        learning_rate=learning_rate,
        network_dim=network_dim,
        network_alpha=network_alpha,
        mixed_precision="bf16"
    )
    args = gen_train_lora_args(output_config=output_config, train_config=train_config, optimizer_config=AdamW8bitConfig())
    subprocess.run(f"accelerate launch --mixed_precision bf16 {sd_scripts_path}/train_network.py {args}", shell=True)


def train_lora_xl(base_path:str,
                  config_file_path:str,
                  model_name:str,
                  max_train_epochs:int,
                  train_batch_size:int,
                  save_every_n_epochs:int=10,
                  learning_rate:float=1e-6,
                  network_dim:int=32,
                  network_alpha:int=32,
                  sampler_config:Optional[SampleConfig] = None,
                  continue_from:Optional[str] = None
                 ):
    print("train called")
    output_config = OutputConfig(
        base_path=base_path,
        model_name=model_name,
        save_every_n_epochs=save_every_n_epochs
    )
    print("output config done!")
    train_config = TrainConfig(
        config_file_path=config_file_path,
        pretrained_model_name_or_path=SDXL,
        max_train_epochs=max_train_epochs,
        train_batch_size=train_batch_size,
        learning_rate=learning_rate,
        network_dim=network_dim,
        network_alpha=network_alpha,
        continue_from=continue_from,
        mixed_precision="bf16"
    )
    print(os.getcwd())
    print(os.getcwdb())

    args = gen_train_lora_args(output_config=output_config, train_config=train_config, sample_config=sampler_config, optimizer_config=AdamW8bitConfig())
    cmd = f"accelerate launch --mixed_precision bf16 {sd_scripts_path}/sdxl_train_network.py {args}"
    print(f"Going to run {cmd}")
    sleep(10)
    if SKIP_PROC:
        print("Skipping training")
        return
    run_cli(cmd)
    print("model trained!")
    return output_config.out_dir

print("Done!")
