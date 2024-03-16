import subprocess
import os
from time import sleep
from apilib.util.env import SLEEP_TIME

def run_cli(args:str):
    print(f"run cli: {args}")
    print(f"sleep {SLEEP_TIME} seconds")
    sleep(SLEEP_TIME)
    try:
        return subprocess.check_output(args, shell=True)
    except subprocess.CalledProcessError as e:
        raise e.stderr

def is_model(name:str)->bool:
    _, ext = os.path.splitext(name)
    return ext in ['.safetensors', '.ckpt']

def get_name(name:str):
    filename, _ = os.path.splitext(name)
    return filename

def get_ext(name:str):
    _, file_extension = os.path.splitext(name)
    return file_extension

def is_ckpt(name:str):
    ext = get_ext(name)
    return ext == ".safetensors" or ext == ".ckpt"

def add_line(file ,key: str, val:str) -> str:
    return file.write(f"{key}: {val}\n")

def trim_map(x: str) -> list[str]:
    return list(map(lambda x: x.strip(), x.split(",")))
