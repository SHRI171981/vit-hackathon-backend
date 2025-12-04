from hf_deployment import upload_model, upload_config, setup_model

# Hugging Face Models

model_1_config = {
        "HF_MODELS_REPO_ID": "shri171981/dl_genai_23f1002153",
        "HF_APP_REPO_ID": "shri171981/medical_chat_generative",
        "HF_LOCAL_MODEL_NAME": "local_models/deberta_base.pth",
        "HF_MODEL_FILE_NAME": "model_microsoft_deberta-v3-base.pth.pth",
        "BASE_MODEL_NAME": "microsoft/deberta-v3-base",
        "LOCAL_FOLDER": "hf_deployment/models/lora_model",
    }

def run_upload(config):
    # upload_model(config)
    # setup_model(config)
    upload_config(config)

if __name__ == "__main__":
    run_upload(model_1_config)
