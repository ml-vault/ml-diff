from typing import Dict
from apilib.train import train_lora_xl
import runpod
from runpod.serverless.utils.rp_validator import validate
from schema import SCHEMAS 
from apilib.download import download_dataset_from_hf
from dotenv import load_dotenv

# .envファイルの内容を読み込見込む
load_dotenv()

def handler(job):
    job_input = job["input"] # Access the input from the request.
    func = job_input["fn"] # Access the function name from the input.
    target_schema = SCHEMAS[func]
    validated_input: Dict = validate(job_input, target_schema) 
    
    if 'errors' in validated_input:
        return {"error": validated_input['errors']}

    if func == "TRAIN_XL_LORA":
        dataset_name = job_input['dataset_repo']
        config_path = download_dataset_from_hf("/_tmp/datasets", dataset_name)
        train_lora_xl(
            "/_tmp/train",
            config_file_path=config_path,
            max_train_epochs=job_input['max_train_epochs'],
            train_batch_size=job_input['train_batch_size'],
            )
        
  # Add your custom code here.
  
    return "Your job results"

runpod.serverless.start({ "handler": handler}) # Required.
