import subprocess

def run_cli(args):
  with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True) as process:
    for line in process.stdout:
      print(line.decode('utf8'))
