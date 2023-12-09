import json
import os
from apilib.train import train_xl_lora_from_datapack
import runpod
from runpod.serverless.utils.rp_validator import validate
from apilib.upload import upload_all_files_to_hf
from schema import SCHEMAS 
from apilib.util.env import TEMP_DIR, HF_USER, R_TOKEN
from mlvault.datapack import DataPack, DataPackLoader


class ValidateError(Exception):
    message: str
    def ___init__(self, message):
        self.message = message
    pass

def handler(job):
    try:
        job_input = job["input"] # Access the input from the request.
        func = job_input["fn"] if "fn" in job_input else ""
        os.makedirs(TEMP_DIR, exist_ok=True)

        # target_schema = SCHEMAS[func]
        # validated_input = validate(job_input, target_schema) 
        
        # if 'errors' in validated_input:
        #     raise ValidateError(validated_input['errors'])

        if func == "TRAIN_XL_LORA":
            repo_id = job_input['dataset_repo']
            datapack = DataPackLoader.load_datapack_from_hf(repo_id, R_TOKEN, TEMP_DIR)
            datapack.export_files(TEMP_DIR, R_TOKEN)
            upload_all_files_to_hf(f"{HF_USER}/{datapack.output.model_name}", TEMP_DIR)
            train_xl_lora_from_datapack(datapack)
            return "Train completed and uploaded to HF"
        else:
            print("loading dynamic datapack")
            datapack = DataPackLoader.load_dynamic_datapack(job_input, TEMP_DIR)
            with open(f'{TEMP_DIR}/input.json', 'w', encoding="utf-8") as f:
                json.dump(job_input, f, indent=2)
            datapack.export_files(TEMP_DIR, R_TOKEN)
            upload_all_files_to_hf(f"{HF_USER}/{datapack.output.model_name}", TEMP_DIR)
            train_xl_lora_from_datapack(datapack)
            return "Train completed and uploaded to HF"

    except ValidateError as e:
        print(e.message)
        return {
            "error": "validate",
            "message":e.message
            }

    except Exception as e:
        return {
            "error": "unknown",
            "message":e.__dict__
        }

        

runpod.serverless.start({ "handler": handler}) # Required.
