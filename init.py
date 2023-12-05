from apilib.download import download_from_hf
from apilib.util.env import MODEL_DIR, R_TOKEN, W_TOKEN
from mlvault.config import set_auth_config

download_from_hf("stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors", MODEL_DIR)
print("donwload SDXL completed!")
set_auth_config(r_token=R_TOKEN, w_token=W_TOKEN)
print("set auth config completed!")
