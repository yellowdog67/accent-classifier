# Accent Classifier

Baseline accent classification model using MFCC feature extraction and a RandomForest classifier.

## Dataset
Audio samples labeled as native / non-native speakers.

## Pipeline
1. Load audio using librosa
2. Extract MFCC features
3. Convert audio into fixed-length feature vectors
4. Train RandomForest classifier
5. Evaluate on 20% held-out validation split

## Results
Validation Accuracy: ~81%

## Tech Stack
- Python
- Librosa
- Scikit-learn
- NumPy
