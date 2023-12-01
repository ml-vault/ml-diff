import os
from typing import Dict
import runpod
from runpod.serverless.utils.rp_validator import validate
from schema import INPUT_SCHEMA, TRAIN_XL_LORA
from apilib.download import download_dataset_from_hf
from dotenv import load_dotenv

# .envファイルの内容を読み込見込む
load_dotenv()
print(os.getenv('HF_USER'))

def handler(job):
    job_input = job["input"] # Access the input from the request.
    func = job_input["fn"] # Access the function name from the input.

    if func == "train_xl_lora":
        validated_input: Dict = validate(job_input, TRAIN_XL_LORA) 
        if 'errors' in validated_input:
            return {"error": validated_input['errors']}
        dataset_name = job_input['dataset']
        download_dataset_from_hf("./_tmp/datasets", dataset_name)
        
  # Add your custom code here.
  
    return "Your job results"

runpod.serverless.start({ "handler": handler}) # Required.
