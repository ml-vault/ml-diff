import json
import os
from apilib.train import train_xl_lora_from_datapack, train_xl_model
from apilib.train.main import resolve_model_name
import runpod
from apilib.util.env import TEMP_DIR, DOWNLOAD_DIR, MODEL_DIR
from mlvault.datapack import DataPack
from mlvault.config import get_w_token
from huggingface_hub import create_repo, upload_folder

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


class ValidateError(Exception):
    message: str
    def ___init__(self, message):
        self.message = message
    pass

def handler(job):
    try:
        job_input = job["input"] # Access the input from the request.
        working_repo = job_input["working_repo"] if "working_repo" in job_input else ""
        repo_dir = job_input['output']['model_name']
        os.environ["WORKING_REPO"] = working_repo
        os.environ["REPO_DIR"] = repo_dir
        local_work_dir = os.path.join(TEMP_DIR, repo_dir)
        os.makedirs(local_work_dir, exist_ok=True)
        print("loading dynamic datapack")
        work_type = job_input['type']
        with open(os.path.join(local_work_dir, "job_input.json"), "w") as f:
            json.dump(job_input, f, indent=4)
        print("wrote input jons")
        resolved = resolve_model_name(job_input['train'].get('pretrained_model_name_or_path')),
        return resolved
        datapack = DataPack(job_input, local_work_dir)
        datapack.export_files()
        create_repo(working_repo,token=get_w_token(), private=True, exist_ok=True)
        upload_folder(folder_path=local_work_dir, repo_id=working_repo, path_in_repo=repo_dir, token=get_w_token(), ignore_patterns=["continue_from", "datasets"])
        if work_type == "TRAIN_XL_LORA":
            train_xl_lora_from_datapack(datapack, job_input)
        elif work_type == "TRAIN_XL":
            train_xl_model(datapack, job_input)
        else:
            raise Exception("unknown work type")
        return "Train completed and uploaded to HF"

    except Exception as e:
        print(e)
        return {
            "error": "unknown",
            "message":e.__dict__
        }

        

runpod.serverless.start({ "handler": handler}) # Required.
