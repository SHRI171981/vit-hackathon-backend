import os
import random
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
INPUT1_LOC = "data/general_classifier/full.csv"
INPUT2_LOC = "data/medical_classifier/full_data.csv"

DATA_LOC = "data/discriminator/full.csv"

TRAIN_LOC = "data/discriminator/train.csv"
VAL_LOC = "data/discriminator/val.csv"
TEST_LOC = "data/discriminator/test.csv"

df1 = pd.read_csv(INPUT1_LOC)
df1['label'] = [0] * df1.shape[0] # non medical

df2 = pd.read_csv(INPUT2_LOC)
df2['label'] = [1] * df2.shape[0] # medical

df = pd.concat((df1, df2), ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv(DATA_LOC, index=False)

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

train_df = pd.concat([X_train, y_train], axis=1)
val_df = pd.concat([X_val, y_val], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_df.to_csv(TRAIN_LOC, index=False)
val_df.to_csv(VAL_LOC, index=False)
test_df.to_csv(TEST_LOC, index=False)