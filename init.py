from apilib.download import download_from_hf
from apilib.util.env import MODEL_DIR, R_TOKEN, W_TOKEN

download_from_hf("stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors", MODEL_DIR)
print("donwload SDXL completed!")
