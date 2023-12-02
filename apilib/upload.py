from apilib.util import is_model
from apilib.util.env import HF_USER, W_TOKEN, R_TOKEN
from huggingface_hub import upload_file, create_repo, file_exists, repo_exists
import os

def determine_repo_id(model_name:str):
    is_existing_repo = repo_exists(f"{HF_USER}/{model_name}", token=R_TOKEN)
    ends_with_version_number = model_name[-2].startswith('-v')
    if is_existing_repo and ends_with_version_number:
        version_number = int(model_name[-1])
        new_version_repo_name =  f"{model_name[:-2]}-v{version_number+1}"
        return determine_repo_id(new_version_repo_name)
    elif is_existing_repo:
        name_with_version = f"{model_name}-v2"
        return determine_repo_id(name_with_version)
    else:
        return f"{HF_USER}/{model_name}"

def upload_models_to_hf(model_name:str, upload_dir:str):
    repo_name = determine_repo_id(model_name)
    repo_exists(repo_name, token=R_TOKEN)
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
    print("model uploaded!")
