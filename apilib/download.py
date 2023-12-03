from typing import Any
from  huggingface_hub import hf_hub_download
from apilib.util.env import HF_USER, R_TOKEN, W_TOKEN
from huggingface_hub import snapshot_download, upload_file, create_repo

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
    from os import makedirs
    import os
    import toml
    from datasets import load_dataset
    from PIL import Image
    import numpy as np
    from huggingface_hub import hf_hub_download
    from tqdm import tqdm 

    user_name = HF_USER
    repo_id = f"{user_name}/{repo_name}"

    r_token = os.getenv('R_TOKEN') #@param {type:"string"
    w_token = os.getenv('W_TOKEN') #@param {type:"string"
    ds:Any = load_dataset(repo_id, split="train", token=r_token)
    dataset_dir = f"{local_dir}/{repo_name}"
    config_file_path = f"{dataset_dir}/config.toml"

    hf_hub_download(repo_id=repo_id, token=r_token, filename="config.toml", repo_type="dataset", local_dir=dataset_dir)
    config = toml.load(config_file_path)
    extension_dict = {}
    general = config['general']
    general_caption_extension = general.get("caption_extension", None)
    for dataset in config['datasets']:
        dataset_caption_extension = dataset.get("caption_extension", None)
        for subset in dataset['subsets']:
            subset_caption_extension = subset.get("caption_extension", None) or dataset_caption_extension or general_caption_extension or '.caption'
            serial_in_config = subset['image_dir']
            subset_dir = f"{dataset_dir}/{serial_in_config}"
            subset['image_dir'] = subset_dir
            extension_dict[serial_in_config] = subset_caption_extension
            makedirs(subset_dir, exist_ok=True)

    for i in tqdm(range(len(ds))):
        data = ds[i]
        file_name=data['file_name']
        image=data['image']
        extension = extension_dict[data['serial']]
        data_subset_dir = f"{dataset_dir}/{data['serial']}"
        base_name = os.path.splitext(file_name)[0]
        to_save_caption_path = f"{data_subset_dir}/{base_name}{extension}"
        to_save_img_path = f"{data_subset_dir}/{file_name}"
        nparr = np.array(image)
        open(to_save_caption_path, 'w').write(data['caption'])
        Image.fromarray(nparr).save(to_save_img_path)
    toml.dump(config, open(config_file_path, 'w'))
    print("dataset downloaded!")
    return f"{local_dir}/{repo_name}/config.toml"
