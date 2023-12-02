from os import getenv
from dotenv import load_dotenv

load_dotenv()

TEMP_DIR = getenv("TEMP_DIR")
DATASET_DIR = f"{TEMP_DIR}/dataset"
TRAIN_DIR = f"{TEMP_DIR}/train"
