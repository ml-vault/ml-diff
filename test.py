from huggingface_hub import hf_hub_download
from mlvault.util import get_r_token
downloaded = hf_hub_download(repo_id="togoron/model_about_girl", filename="about_girl_2/config.toml", local_dir="/tmp", token=get_r_token(), local_dir_use_symlinks=False)
print(downloaded)
