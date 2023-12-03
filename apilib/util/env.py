from os import getenv
from dotenv import load_dotenv

load_dotenv()

TEMP_DIR = getenv("TEMP_DIR")
DATASET_DIR = f"{TEMP_DIR}/dataset"
TRAIN_DIR = f"{TEMP_DIR}/train"

R_TOKEN = getenv("R_TOKEN")
W_TOKEN = getenv("W_TOKEN")
HF_USER = getenv("HF_USER")
SKIP_PROC = getenv("SKIP_PROC")
MODEL_DIR = getenv("MODEL_DIR")

SDXL = f"{MODEL_DIR}/sd_xl_base_1.0.safetensors"
