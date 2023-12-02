from apilib.util import is_model
from apilib.util.env import HF_USER, W_TOKEN
from huggingface_hub import upload_file, create_repo, file_exists
import os

def upload_models_to_hf(model_name:str, upload_dir:str):
    repo_name = f"{HF_USER}/{model_name}"
    create_repo(repo_name, token=W_TOKEN, private=True, exist_ok=True)
    file_list = list(os.listdir(upload_dir))
    models = list(filter(lambda file: is_model(file) == True, file_list))
    
    for file_path in models:
      base_name = os.path.basename(file_path)
      file_full_path=f"{upload_dir}/{file_path}"
      if not file_exists(model_name, base_name):
        upload_file(path_or_fileobj=file_full_path, path_in_repo=base_name, repo_id=repo_name, token=W_TOKEN)
      else:
        print(f"file existing: {base_name}")
    print("done!")
