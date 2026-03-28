import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------
# Paths
# -----------------------------
CSV_PATH = "data/labels.csv"
IMAGE_DIR = "data/train"
OUTPUT_DIR = "data"

TRAIN_DIR = os.path.join(OUTPUT_DIR, "train_split")
VAL_DIR = os.path.join(OUTPUT_DIR, "valid")

# -----------------------------
# Create folders
# -----------------------------
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# -----------------------------
# Load CSV
# -----------------------------
print("📄 Reading labels...")
df = pd.read_csv(CSV_PATH)

# -----------------------------
# Train-Validation Split
# -----------------------------
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['breed']   # keep class balance
)

print(f"✅ Train size: {len(train_df)}")
print(f"✅ Validation size: {len(val_df)}")

# -----------------------------
# Function to move images
# -----------------------------
def organize_data(dataframe, output_folder):
    for _, row in dataframe.iterrows():
        image_id = row['id']
        breed = row['breed']

        src = os.path.join(IMAGE_DIR, image_id + ".jpg")
        dst_folder = os.path.join(output_folder, breed)

        os.makedirs(dst_folder, exist_ok=True)

        dst = os.path.join(dst_folder, image_id + ".jpg")

        if os.path.exists(src):
            shutil.copy(src, dst)


# -----------------------------
# Organize Train Data
# -----------------------------
print("📂 Organizing training data...")
organize_data(train_df, TRAIN_DIR)

# -----------------------------
# Organize Validation Data
# -----------------------------
print("📂 Organizing validation data...")
organize_data(val_df, VAL_DIR)

print("🎉 Data preprocessing complete!")