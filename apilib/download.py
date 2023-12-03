from  huggingface_hub import hf_hub_download
from apilib.datapack import DataPackLoader
from apilib.util.env import HF_USER, R_TOKEN, W_TOKEN
from huggingface_hub import snapshot_download, upload_file, create_repo
from os.path import join as join_path

sd_lora_dir = "/workspace/sd/stable-diffusion-webui/models/Lora"
sd_model_dir='/workspace/sd/stable-diffusion-webui/models/Stable-diffusion'

def download_from_hf(repo_id:str, filename:str, local_dir:str):
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir, force_download=True, local_dir_use_symlinks=False)

def ask(msg:str):
    print(msg)
    return input()

def download_lora_from_hf(hf_repo_name=""):
    repo_name=hf_repo_name or ask("hf lora repo name?")
    repo_id=f"{HF_USER}/{repo_name}"
    download_dir = f"{sd_lora_dir}/{repo_name}"
    snapshot_download(repo_id=repo_id, local_dir=download_dir, token=R_TOKEN, cache_dir="/workspace/hub-cache")
    
def download_ckpt_from_civit_ai(version:str, ckpt_model_id=""):
    model_id= ckpt_model_id or ask("ckpt model id?")
    down_dir=f"{sd_model_dir}/{version}"
    # !wget -P $down_dir https://civitai.com/api/download/models/$model_id --content-disposition

def download_lora_from_civit_ai(lora_model_id=""):
    model_id=lora_model_id or ask("lora model id?")
    # !wget -P $sd_lora_dir https://civitai.com/api/download/models/$model_id --content-disposition
    
def download_dataset_from_hf(local_dir:str, repo_name:str):
    pack_dir = f"{local_dir}/{repo_name}"
    repo_id = f"{HF_USER}/{repo_name}"
    datapack = DataPackLoader.load_datapack_from_hf(repo_id, R_TOKEN)
    dataset_dir = join_path(pack_dir, "dataset")
    hf_hub_download(repo_id=repo_id, token=R_TOKEN, filename="config.yml", repo_type="dataset", local_dir=local_dir)
    datapack.load_files(dataset_dir, R_TOKEN)
    print("dataset downloaded!")
    return f"{local_dir}/{repo_name}/config.toml"
