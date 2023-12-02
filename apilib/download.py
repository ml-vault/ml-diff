from typing import Any
from dotenv import load_dotenv
from  huggingface_hub import hf_hub_download
import os
load_dotenv()

hf_username= os.getenv('HF_USER')
r_token = os.getenv('R_TOKEN')
w_token = os.getenv('W_TOKEN')

sd_lora_dir = "/workspace/sd/stable-diffusion-webui/models/Lora"
sd_model_dir='/workspace/sd/stable-diffusion-webui/models/Stable-diffusion'

from huggingface_hub import snapshot_download, notebook_login, upload_file, create_repo

def download_from_hf(repo_id:str, filename:str, local_dir:str):
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir, force_download=True, local_dir_use_symlinks=False)

def ask(msg:str):
    print(msg)
    return input()

def is_model(name:str)->bool:
    file_name, ext = os.path.splitext(name)
    return ext in ['.safetensors', '.ckpt']

def download_lora_from_hf(hf_repo_name=""):
    repo_name=hf_repo_name or ask("hf lora repo name?")
    repo_id=f"{hf_username}/{repo_name}"
    download_dir = f"{sd_lora_dir}/{repo_name}"
    snapshot_download(repo_id=repo_id, local_dir=download_dir, token=r_token, cache_dir="/workspace/hub-cache")
    
def download_ckpt_from_civit_ai(version:str, ckpt_model_id=""):
    model_id= ckpt_model_id or ask("ckpt model id?")
    down_dir=f"{sd_model_dir}/{version}"
    # !wget -P $down_dir https://civitai.com/api/download/models/$model_id --content-disposition

def download_lora_from_civit_ai(lora_model_id=""):
    model_id=lora_model_id or ask("lora model id?")
    # !wget -P $sd_lora_dir https://civitai.com/api/download/models/$model_id --content-disposition
    
def upload_models_to_hf(model_name:str, upload_dir:str):
    repo_name = f"{hf_username}/{model_name}"
    create_repo(repo_name, token=w_token, private=True, exist_ok=True)
    file_list = list(os.listdir(upload_dir))
    models = list(filter(lambda file: is_model(file) == True, file_list))
    
    for file_path in models:
      base_name = os.path.basename(file_path)
      file_full_path=f"{upload_dir}/{file_path}"
    #   if not file_exists(model_name, base_name):
      upload_file(path_or_fileobj=file_full_path, path_in_repo=base_name, repo_id=repo_name, token=w_token)
    #   else:
    #     print(f"file existing: {base_name}")
    print("done!")

def download_dataset_from_hf(local_dir:str, repo_name:str):
    from os import makedirs
    import os
    import toml
    from datasets import load_dataset
    from PIL import Image
    import numpy as np
    from huggingface_hub import hf_hub_download
    from tqdm import tqdm 

    user_name = hf_username
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
