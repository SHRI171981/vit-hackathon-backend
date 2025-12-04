import os
import random
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
DATA_LOC = "data/general_classifier/full.csv"
TRAIN_LOC = "data/general_classifier/train.csv"
VAL_LOC = "data/general_classifier/val.csv"
TEST_LOC = "data/general_classifier/test.csv"

# We want 15k Safe + 15k Unsafe = 30k Total High-Quality Rows
TARGET_PER_CLASS = 15000 

def get_unsafe_data(limit):
    """
    Collects strictly harmful data:
    1. ToxicChat (Real User Toxicity)
    2. Jigsaw (Explicit Threats)
    3. Anthropic (Red Team Attacks)
    """
    print(f"--- Collecting {limit} UNSAFE (Label=1) examples ---")
    rows = []
    
    # 1. ToxicChat (High Quality)
    try:
        print(" > Loading ToxicChat...")
        ds = load_dataset("lmsys/toxic-chat", "toxic-chat-0124", split="train")
        for x in ds:
            if x['toxicity'] == 1:
                rows.append(x['user_input'])
    except Exception as e: print(f"ToxicChat Error: {e}")

    # 2. Jigsaw (Civil Comments) - FILTERED FOR THREATS
    # We only want threats to catch "I want to kill"
    if len(rows) < limit:
        try:
            print(" > Loading Jigsaw (Threats)...")
            ds = load_dataset("google/civil_comments", split="train", streaming=True)
            for x in ds:
                if len(rows) >= limit: break
                # Strict filter: Threat OR Severe Toxicity
                if x['threat'] > 0.5 or x['severe_toxicity'] > 0.5:
                    rows.append(x['text'])
        except Exception as e: print(f"Jigsaw Error: {e}")

    # 3. Anthropic Red Team (If we still need more)
    if len(rows) < limit:
        try:
            print(" > Loading Anthropic Red Team...")
            ds = load_dataset("Anthropic/hh-rlhf", data_dir="red-team-attempts", split="train")
            for x in ds:
                if len(rows) >= limit: break
                if "Human: " in x['transcript']:
                    prompt = x['transcript'].split("Human: ")[1].split("Assistant:")[0].strip()
                    rows.append(prompt)
        except Exception as e: print(f"Anthropic Error: {e}")
        
    # Truncate to limit
    return rows[:limit]

def get_safe_data(limit):
    """
    Collects strictly safe data:
    1. XSTest (Hard Negatives - Critical for Medical)
    2. Alpaca (Normal Instructions)
    """
    print(f"--- Collecting {limit} SAFE (Label=0) examples ---")
    rows = []

    # 1. XSTest (Hard Negatives)
    try:
        print(" > Loading XSTest (Safe but tricky)...")
        ds = load_dataset("allenai/xstest-response", split="train")
        for x in ds:
            rows.append(x['prompt'])
    except Exception as e: print(f"XSTest Error: {e}")

    # 2. Alpaca (Normal User Queries)
    if len(rows) < limit:
        try:
            print(" > Loading Alpaca (Normal Instructions)...")
            ds = load_dataset("tatsu-lab/alpaca", split="train")
            for x in ds:
                if len(rows) >= limit: break
                # Combine instruction + input
                text = x['instruction']
                if x['input']:
                    text += " " + x['input']
                rows.append(text)
        except Exception as e: print(f"Alpaca Error: {e}")
        
    # Truncate to limit
    return rows[:limit]

# --- MAIN EXECUTION ---

# Ensure directories exist
os.makedirs(os.path.dirname(DATA_LOC), exist_ok=True)

# 1. Gather Data
unsafe_texts = get_unsafe_data(TARGET_PER_CLASS)
safe_texts = get_safe_data(TARGET_PER_CLASS)

print(f"\nFinal Counts -> Unsafe: {len(unsafe_texts)} | Safe: {len(safe_texts)}")

# 2. Create DataFrames
df_unsafe = pd.DataFrame({'text': unsafe_texts, 'label': 1})
df_safe = pd.DataFrame({'text': safe_texts, 'label': 0})

# 3. Combine and Shuffle
df = pd.concat([df_unsafe, df_safe], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 4. Save Master File
df.to_csv(DATA_LOC, index=False)
print(f"Saved Full Dataset to {DATA_LOC}")

# 5. Split (Train: 80%, Val: 10%, Test: 10%)
# First split Train (80%) vs Temp (20%)
X_train, X_temp, y_train, y_temp = train_test_split(
    df['text'], df['label'], 
    test_size=0.2, 
    stratify=df['label'], # CRITICAL: Keeps the 50/50 balance in splits
    random_state=42
)

# Second split Temp (20%) into Val (10%) and Test (10%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, 
    test_size=0.5, 
    stratify=y_temp, 
    random_state=42
)

# 6. Save Splits
train_df = pd.concat([X_train, y_train], axis=1)
val_df = pd.concat([X_val, y_val], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_df.to_csv(TRAIN_LOC, index=False)
val_df.to_csv(VAL_LOC, index=False)
test_df.to_csv(TEST_LOC, index=False)

print("\n--- DONE ---")
print(f"Train: {len(train_df)} rows")
print(f"Val:   {len(val_df)} rows")
print(f"Test:  {len(test_df)} rows")