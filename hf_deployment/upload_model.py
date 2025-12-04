from huggingface_hub import HfApi, login
from dotenv import load_dotenv
import os

def upload_model(config):
    # Load environment variables from .env file
    load_dotenv()
    login(token=os.getenv("HF_TOKEN"))

    api = HfApi()

    # 2. Define names
    repo_id = config['HF_MODELS_REPO_ID'] # Repo name on the Hub
    local_file = config['HF_LOCAL_MODEL_NAME']

    # 3. Create the repo (if it doesn't exist yet)
    api.create_repo(repo_id=repo_id, exist_ok=True)

    # 4. Upload the file
    api.upload_file(
        path_or_fileobj=local_file,
        path_in_repo=config['HF_MODEL_FILE_NAME'], # Name on the Hub
        repo_id=repo_id
    )

    print("1. Uploaded model")