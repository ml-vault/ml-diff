from setuptools import setup, find_packages

def requirements_from_file(file_name):
    return open(file_name).read().splitlines()

 
setup(name = "apilib", 
      packages = find_packages(),
      version="0.0.1",
      install_requires=requirements_from_file("requirements.txt"),
      )
