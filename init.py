import os
from apilib.download import download_from_hf
from apilib.util.env import MODEL_DIR, DOWNLOAD_DIR, TEMP_DIR

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
download_from_hf("stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors", MODEL_DIR)
print("donwload SDXL completed!")
