import pandas as pd
import os
import requests

file_name = "Nativity Assessmet Audio Dataset.xlsx"

# Read training sheet
df_train = pd.read_excel(file_name, sheet_name="Training Dataset")

print("Training data loaded successfully!")

# Create folders
os.makedirs("data/train/native", exist_ok=True)
os.makedirs("data/train/non_native", exist_ok=True)

for index, row in df_train.iterrows():
    url = row["audio_url"]
    label = row["nativity_status"]
    file_id = row["dp_id"]

    # Decide folder
    if label == "Native":
        save_path = f"data/train/native/{file_id}.mp3"
    else:
        save_path = f"data/train/non_native/{file_id}.mp3"

    # Download file
    try:
        response = requests.get(url)
        with open(save_path, "wb") as f:
            f.write(response.content)

        print(f"Downloaded {file_id}")

    except Exception as e:
        print(f"Failed to download {file_id}: {e}")
