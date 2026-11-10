# Medical Chat Guard: Data and Model Pipeline

## Problem statement

A medical chatbot has two jobs before it answers: it must recognise that a question is actually about health, and it must refuse harmful or abusive input. A generic safety filter fails here, because medical questions often sound alarming ("What dose of this drug is dangerous?") and get wrongly blocked.

## Solution

The data preparation and deployment pipeline for a set of small text classifiers that sit in front of a medical chatbot:

- **Safety classifier (safe vs unsafe).** Trained on 15,000 balanced examples per class. Unsafe text comes from ToxicChat, threat-labelled Civil Comments and Anthropic red-team prompts. Safe text comes from XSTest (deliberately hard negatives that sound risky but are fine) and Alpaca instructions.
- **Medical classifier** built from a labelled medical dataset.
- **Discriminator (medical vs non-medical)**, a fine-tuned DistilBERT that routes each question, built by combining the general and medical datasets with balanced 80/10/10 stratified splits.

Trained weights are uploaded to the Hugging Face Hub and packaged into deployable model folders with the tokenizer.

## Tech stack

Python, PyTorch, Hugging Face Transformers and Datasets, DistilBERT, DeBERTa, scikit-learn, Pandas, Hugging Face Hub.

## Features

- Reproducible dataset builders that pull from public Hugging Face datasets and write train, validation and test CSVs
- Stratified splits that keep classes balanced
- Hard-negative examples so medical questions are not wrongly flagged as unsafe
- Hugging Face deployment helpers: upload weights, rebuild the model architecture from a saved state dict, save model and tokenizer in the standard format, and upload the configuration
- Central configuration for repositories, file names and the number of classes

## Future steps

- Add the model training scripts and evaluation reports
- Serve the classifiers behind an API
- Evaluate on a held-out set of tricky medical questions
- Add a confidence threshold with a human-review fallback

## Getting started

```bash
pip install torch transformers datasets scikit-learn pandas huggingface_hub

python scripts/prep_general_classifier_data.py
python scripts/prep_medical_classifier_data.py
python scripts/prep_discriminator_data.py
```

Hugging Face upload helpers are in `hf_deployment/`, and repository names are set in `config.py`.
