from huggingface_hub import HfApi, login
from dotenv import load_dotenv
import os


def upload_config(config):
    load_dotenv()
    login(token=os.getenv("HF_TOKEN"))

    api = HfApi()

    api.create_repo(repo_id=config['HF_APP_REPO_ID'], exist_ok=True)

    api.upload_folder(
        folder_path=config['LOCAL_FOLDER'],
        repo_id=config['HF_APP_REPO_ID'],
        path_in_repo="."
    )

    print("3. Uploaded final config")