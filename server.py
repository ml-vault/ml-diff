import os
from apilib.train import train_xl_lora_from_datapack, train_xl_model
import runpod
from apilib.upload import upload_all_files_to_hf
from apilib.util.env import TEMP_DIR, HF_USER
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

        if func == "TRAIN_XL_LORA":
            repo_id = job_input['dataset_repo']
            datapack = DataPackLoader.load_datapack_from_hf(repo_id, TEMP_DIR)
            datapack.export_files()
            upload_all_files_to_hf(f"{HF_USER}/{datapack.output.model_name}", TEMP_DIR)
            train_xl_lora_from_datapack(datapack)
            return "Train completed and uploaded to HF"
        else:
            print("loading dynamic datapack")
            work_type = job_input['type']
            datapack = DataPack(job_input, TEMP_DIR)
            datapack.export_files()
            upload_all_files_to_hf(f"{HF_USER}/{datapack.output.model_name}", TEMP_DIR)
            if work_type == "TRAIN_XL_LORA":
                train_xl_lora_from_datapack(datapack)
            elif work_type == "TRAIN_XL":
                train_xl_model(datapack)
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
