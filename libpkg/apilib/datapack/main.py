import os
from os.path import join as join_path
from os.path import isfile
from datasets import Dataset
from PIL import Image
import numpy as np
from huggingface_hub import upload_file
import toml
import yaml
from datasets.load import load_dataset
from huggingface_hub.file_download import hf_hub_download
from tqdm import tqdm
from datasets.dataset_dict import IterableDatasetDict
from typing import Any

def to_optional_dict(d:Any, keys:list[str]):
    output = {}
    for key in keys:
        val = d.__dict__[key] if hasattr(d, key) else None
        if(val):
            output[key] = val
    return output

def get_ext(file_name: str):
    return os.path.splitext(file_name)[1].lower()

def list_dir_in_ext(path: str, ext: str):
    files = os.listdir(path)
    filtered = filter(lambda file: file.endswith(ext), files)
    return list(filtered)

imgs = ['.png', '.jpg', '.jpeg', '.bmp', '.webp']

def list_dir_in_img(path:str):
    files = os.listdir(path)
    filtered = filter(lambda file: get_ext(file) in imgs, files)
    return list(filtered)   


class DataTray:
    imgs: list = []
    caption: list[str] = []
    div: list[str] = []
    file_name: list[str] = []
    caption_extension:list[str] = []

    def add(self, div: str, img_file_name: str, image: Image.Image, caption: str, caption_extension:str):
        self.div.append(div)
        self.file_name.append(img_file_name)
        self.imgs.append(image)
        self.caption.append(caption)
        self.caption_extension.append(caption_extension)

    def to_dataset(self):
        ds = Dataset.from_dict(
            {
                "div": self.div,
                "image": self.imgs,
                "caption": self.caption,
                "file_name": self.file_name,
                "caption_extension": self.caption_extension
            }
        )
        return ds
    
    def push_to_hub(self, repo_id:str, w_token:str):
        self.to_dataset().push_to_hub(repo_id, token=w_token, private=True)


class SubsetConfig:

    def __init__(self, name:str, config_input:dict) -> None:
        self.path = config_input["path"]
        self.name = name
        if "class_tokens" in config_input:
            self.class_tokens = config_input["class_tokens"]
        if "is_reg" in config_input:
            self.is_reg = config_input["is_reg"]
        if "caption_extension" in config_input:
            self.caption_extension = config_input["caption_extension"]
        if "keep_tokens" in config_input:
            self.keep_tokens = config_input["keep_tokens"]
        if "num_repeats" in config_input:
            self.num_repeats = config_input["num_repeats"]
        if "shuffle_caption" in config_input:
            self.shuffle_caption = config_input["shuffle_caption"]
        pass
    
    def to_toml_dict(self, dataset_dir:str):
        toml_dict = to_optional_dict(self, ["caption_extension", "keep_tokens", "num_repeats", "shuffle_caption", "class_tokens", "is_reg"])
        toml_dict["image_dir"] = join_path(dataset_dir, self.name)
        return toml_dict


class DatasetConfig:
    subsets: dict[str, SubsetConfig] = {}

    def __init__(self, name:str, config_input:dict) -> None:
        self.name = name
        if "resolution" in config_input:
            self.resolution = config_input["resolution"]
        if "caption_extension" in config_input:
            self.caption_extension = config_input["caption_extension"]
        if "keep_tokens" in config_input:
            self.keep_tokens = config_input["keep_tokens"]
        if "num_repeats" in config_input:
            self.num_repeats = config_input["num_repeats"]
        if "shuffle_caption" in config_input:
            self.shuffle_caption = config_input["shuffle_caption"]
        for dataset_key in config_input['subsets']:
            config = { **config_input, **config_input['subsets'][dataset_key]}
            self.subsets[dataset_key] = SubsetConfig(dataset_key, config)
        pass

    def to_dict(self):
        dict_subsets = {}
        for subset_key in self.subsets:
            dict_subsets[subset_key] = self.subsets[subset_key].__dict__
        return {
            "caption_extension": self.caption_extension,
            "name": self.name,
            "subsets": dict_subsets
        }
    def to_toml_dict(self, dataset_dir:str):
        toml_dict = to_optional_dict(self, ["resolution"])
        toml_dict["subsets"] = []
        for subset_key in self.subsets:
            toml_dict["subsets"].append(self.subsets[subset_key].to_toml_dict(join_path(dataset_dir, self.name)))
        return toml_dict


class InputConfig:
    datasets: dict[str, DatasetConfig] = {}

    def __init__(self, config_input:dict) -> None:
        self.repo_id = config_input["repo_id"]
        if "resolution" in config_input:
            self.resolution = config_input["resolution"]
        if "caption_extension" in config_input:
            self.caption_extension = config_input["caption_extension"]
        if "keep_tokens" in config_input:
            self.keep_tokens = config_input["keep_tokens"]
        if "num_repeats" in config_input:
            self.num_repeats = config_input["num_repeats"]
        if "shuffle_caption" in config_input:
            self.shuffle_caption = config_input["shuffle_caption"]
        for dataset_key in config_input['datasets']:
            config = { **config_input, **config_input['datasets'][dataset_key]}
            self.datasets[dataset_key] = DatasetConfig(dataset_key, config)
        pass

    def to_dict(self):
        dict_datasets = {}
        for dataset_key in self.datasets:
            dict_datasets[dataset_key] = self.datasets[dataset_key].to_dict()
        return {
            "repo_id": self.repo_id,
            "datasets": dict_datasets
        }
    
    def to_toml_dict(self, datasets_dir:str):
        toml_dict = {"general":{"enable_bucket":True}, "datasets":[]}
        for dataset_key in self.datasets:
            toml_dict["datasets"].append(self.datasets[dataset_key].to_toml_dict(datasets_dir))
        return toml_dict


class OutputConfig:

    def __init__(self, config_input:dict) -> None:
        self.model_name = config_input["model_name"]
        self.save_every_n_epochs = config_input["save_every_n_epochs"]
        self.save_model_as = config_input["save_model_as"]
        pass 

class TrainConfig:
    def __init__(self, confing_input:dict) -> None:
        self.learning_rate = confing_input["learning_rate"]
        self.train_batch_size = confing_input["train_batch_size"]
        self.network_dim = confing_input["network_dim"]
        self.network_alpha = confing_input["network_alpha"]
        self.max_train_epochs = confing_input["max_train_epochs"]
        self.base_model = confing_input["base_model"]
        self.optimizer = confing_input["optimizer"]
        pass

class SampleConfig:
    def __init__(self, config_input) -> None:
        self.sample_every_n_epochs = config_input["sample_every_n_epochs"]
        self.prompts = config_input["prompts"]
        self.sampler = config_input["sampler"]
        pass

class DataPack:

    def __init__(self, config_file_path:str):
        config = yaml.load(open(config_file_path, 'r'), Loader=yaml.FullLoader)
        self.config_file_path = config_file_path
        self.input = InputConfig(config["input"])
        self.output = OutputConfig(config["output"])
        self.train = TrainConfig(config["train"])
        self.sample = SampleConfig(config["sample"])
    
    def to_dict(self):
        return {
            "input": self.input.to_dict()
        }
    
    def to_data_tray(self):
        data_tray = DataTray()
        for dataset_key in self.input.datasets:
            for subset_key in self.input.datasets[dataset_key].subsets:
                subset = self.input.datasets[dataset_key].subsets[subset_key]
                extension = subset.caption_extension
                src_dir = subset.path
                img_names = list_dir_in_img(src_dir)
                caption = []
                for img_name in img_names:
                    name, _ = os.path.splitext(img_name)
                    tag_name = f"{name}{extension}"
                    caption = ""
                    img_path = f"{src_dir}/{img_name}"
                    img = Image.open(img_path)
                    if tag_name and  isfile(f"{src_dir}/{tag_name}"):
                        caption = open(f"{src_dir}/{tag_name}", "r").read()
                    data_tray.add(
                        div=f"{dataset_key}/{subset_key}", image=img, caption=caption, img_file_name=img_name, caption_extension=extension
                    )
        return data_tray
    
    def push_to_hub(self, w_token:str):
        self.to_data_tray().push_to_hub(self.input.repo_id, w_token)
        upload_file(
            repo_id=self.input.repo_id,
            path_or_fileobj=self.config_file_path,
            path_in_repo="config.yml",
            token=w_token,
            repo_type="dataset",
        )
    
    def export_files(self, base_dir:str, r_token:str):
        print("start exporting files!")
        hf_hub_download(repo_id=self.input.repo_id, filename="config.yml", repo_type="dataset", local_dir=base_dir, token=r_token)
        dataset_dir = f"{base_dir}/datasets"
        self.__export_datasets(dataset_dir, r_token)
        self.__write_sample_prompt(base_dir)
        self.__write_toml(base_dir)
    
    def __write_sample_prompt(self, base_dir:str):
        sample_prompt:list[str] = self.sample.prompts
        sample_prompt_path = f"{base_dir}/sample.txt"
        open(sample_prompt_path, "w").write("\n".join(sample_prompt))
        print("sample prompt written!")

    def __write_toml(self, base_dir:str):
        toml_dict = self.input.to_toml_dict(join_path(base_dir, "datasets"))
        toml_path = f"{base_dir}/config.toml"
        toml.dump(toml_dict, open(toml_path, "w"))
        self.toml_path = toml_path
        print("toml written!")
    
    def __export_datasets(self, dataset_dir:str, r_token:str):
        repo_id = self.input.repo_id
        dataset: IterableDatasetDict = load_dataset(repo_id, split="train", token=r_token) # type: ignore
        num_rows:int = len(dataset)
        for i in tqdm(range(num_rows)):
            data = dataset[i]
            div = data['div']
            subset_dir = f"{dataset_dir}/{div}"
            os.makedirs(subset_dir, exist_ok=True)
            file_name = data['file_name']
            base_name = os.path.splitext(file_name)[0]
            extension = data['caption_extension']
            caption = data['caption']
            image = data['image']
            to_save_img_path = f"{subset_dir}/{file_name}"
            to_save_caption_path = f"{subset_dir}/{base_name}{extension}"
            caption = data['caption']
            open(to_save_caption_path, 'w').write(caption)
            nparr = np.array(image)
            Image.fromarray(nparr).save(to_save_img_path)
        print("datasets exported!")

class DataPackLoader:
    @staticmethod
    def load_datapack_from_hf(repo_id:str, r_token:str, base_dir:str) -> DataPack:
        hf_hub_download(repo_id=repo_id, filename="config.yml", repo_type="dataset", local_dir=base_dir, token=r_token)
        config_file_path = f"{base_dir}/config.yml"
        return DataPack(config_file_path)

