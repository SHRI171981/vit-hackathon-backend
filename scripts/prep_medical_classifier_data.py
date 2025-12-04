import pandas as pd
from sklearn.model_selection import train_test_split

DATA_LOC = "data/medical_classifier/full_data.csv"

TRAIN_LOC = "data/medical_classifier/train.csv"
VAL_LOC = "data/medical_classifier/val.csv"
TEST_LOC = "data/medical_classifier/test.csv"

df = pd.read_csv(DATA_LOC)
X, y = df['text'], df['label']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, shuffle=True, random_state=42)

train_df = pd.concat((X_train, y_train), axis=1)
val_df = pd.concat((X_val, y_val), axis=1)
test_df = pd.concat((X_test, y_test), axis=1)

train_df.to_csv(TRAIN_LOC, index = False)
val_df.to_csv(VAL_LOC, index = False)
test_df.to_csv(TEST_LOC, index = False)