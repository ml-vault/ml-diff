from typing import Dict
from apilib.train import train_lora_xl
import runpod
from runpod.serverless.utils.rp_validator import validate
from schema import SCHEMAS 
from apilib.download import download_dataset_from_hf
from dotenv import load_dotenv
from apilib.util.dir import DATASET_DIR, TRAIN_DIR

# .envファイルの内容を読み込見込む
load_dotenv()

class ValidateError(Exception):
    message: str
    def ___init__(self, message):
        self.message = message
    pass

def handler(job):
    try:
        job_input = job["input"] # Access the input from the request.
        func = job_input["fn"] # Access the function name from the input.
        target_schema = SCHEMAS[func]
        validated_input: Dict = validate(job_input, target_schema) 
        
        if 'errors' in validated_input:
            raise ValidateError(validated_input['errors'])

        if func == "TRAIN_XL_LORA":
            dataset_name = job_input['dataset_repo']
            config_path = download_dataset_from_hf(DATASET_DIR, dataset_name)
            train_lora_xl(
                TRAIN_DIR,
                config_file_path=config_path,
                max_train_epochs=validated_input['max_train_epochs'],
                train_batch_size=validated_input['train_batch_size'],
                model_name=validated_input['model_name'],
                save_every_n_epochs=validated_input['save_every_n_epochs'],
                learning_rate=validated_input['learning_rate'],
                network_dim=validated_input["network_dim"],
                network_alpha=validated_input["network_alpha"]
                )
            
    # Add your custom code here.
    
        return "Your job results"

    except ValidateError as e:
        print(e.message)
        return {
            "error": "validate",
            "message":e.message
            }

    except Exception as e:
        print(e)
        return "Has error"

        

runpod.serverless.start({ "handler": handler}) # Required.
