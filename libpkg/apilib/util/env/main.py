from os import getenv
from dotenv import load_dotenv

load_dotenv()

TEMP_DIR = getenv("TEMP_DIR", "/runpod-volume/_tmp")
SLEEP_TIME = int(getenv("SLEEP_TIME", 0))
DATASET_DIR = f"{TEMP_DIR}/dataset"
TRAIN_DIR = f"{TEMP_DIR}/train"

R_TOKEN = getenv("R_TOKEN", "")
W_TOKEN = getenv("W_TOKEN", "")
HF_USER = getenv("HF_USER", "")
SKIP_PROC = getenv("SKIP_PROC")
MODEL_DIR = getenv("MODEL_DIR", "/root/_models")
DOWNLOAD_DIR = getenv("DOWNLOAD_DIR", "/root/_downloads")

SDXL = f"{MODEL_DIR}/sd_xl_base_1.0.safetensors"
