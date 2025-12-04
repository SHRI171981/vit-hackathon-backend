import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from config import NUM_CLASSES
from huggingface_hub import hf_hub_download

def setup_model(config):

    # 2. Empty architecture
    model_config = AutoConfig.from_pretrained(config['BASE_MODEL_NAME'], num_labels = NUM_CLASSES)
    model = AutoModelForSequenceClassification.from_config(model_config)

    # 3. Downloading weight files
    FILE_PATH = hf_hub_download(repo_id=config['HF_MODELS_REPO_ID'], filename=config['HF_MODEL_FILE_NAME'])

    # 4. Loading weights into that architecture
    state_dict = torch.load(FILE_PATH, map_location="cpu")
    model.load_state_dict(state_dict, strict = True)

    # 5. Save it correctly for Hugging Face
    model.save_pretrained(config['LOCAL_FOLDER'])

    # 6. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['BASE_MODEL_NAME'])
    tokenizer.save_pretrained(config['LOCAL_FOLDER'])

    print("2. Setup completed")