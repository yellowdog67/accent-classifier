import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from feature_extraction import extract_features

X = []
y = []

print("Starting feature extraction...")

# Load native samples
native_folder = "data/train/native"
for file in os.listdir(native_folder):
    file_path = os.path.join(native_folder, file)
    try:
        features = extract_features(file_path)
        X.append(features)
        y.append(1)
    except Exception:
        print("Skipping native file:", file)



# Load non-native samples
non_native_folder = "data/train/non_native"
for file in os.listdir(non_native_folder):
    file_path = os.path.join(non_native_folder, file)
    try:
        features = extract_features(file_path)
        X.append(features)
        y.append(0)
    except Exception:
        print("Skipping non-native file:", file)


X = np.array(X)
y = np.array(y)

print("Feature extraction complete.")
print("Total samples:", len(X))

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_val)

print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))
