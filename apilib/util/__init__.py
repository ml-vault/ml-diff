import subprocess

def run_cli(args:str):
    subprocess.call(args.split(), shell=True)
