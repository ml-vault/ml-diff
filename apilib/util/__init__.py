import subprocess

def run_cli(args):
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    while proc.poll() is None:
        for line in proc.stdout:
            print(line.decode('utf8'))
