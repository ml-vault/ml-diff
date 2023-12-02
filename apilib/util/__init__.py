import subprocess
import os

def run_cli(args:str):
    subprocess.call(args.split())

def is_model(name:str)->bool:
    file_name, ext = os.path.splitext(name)
    return ext in ['.safetensors', '.ckpt']
