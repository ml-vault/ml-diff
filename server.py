from apilib.train import train_xl_lora_from_datapack
import runpod
from runpod.serverless.utils.rp_validator import validate
from apilib.upload import upload_all_files_to_hf
from schema import SCHEMAS 
from apilib.util.env import TEMP_DIR, HF_USER, R_TOKEN
from apilib.datapack import DataPackLoader, DataPack


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
        validated_input = validate(job_input, target_schema) 
        
        if 'errors' in validated_input:
            raise ValidateError(validated_input['errors'])

        if func == "TRAIN_XL_LORA":
            repo_id = job_input['dataset_repo']
            datapack:DataPack = DataPackLoader.load_datapack_from_hf(repo_id, R_TOKEN, TEMP_DIR)
            datapack.export_files(TEMP_DIR, R_TOKEN)
            train_xl_lora_from_datapack(datapack)
            upload_all_files_to_hf(f"{HF_USER}/{datapack.output.model_name}", TEMP_DIR)
            return "Train completed and uploaded to HF"

        return "Unknown function"

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
