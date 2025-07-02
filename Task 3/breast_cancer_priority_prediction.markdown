# Breast Cancer Priority Prediction

This notebook preprocesses the Kaggle Breast Cancer Dataset, extracts features from PNG images, trains a Random Forest classifier to predict issue priority (high/low), and evaluates performance using accuracy and F1-score.

## Setup
Install required libraries and import dependencies.

```python
!pip install opencv-python scikit-image scikit-learn pandas numpy matplotlib seaborn
```

```python
import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
```

## Load and Preprocess Data
Assume the dataset is stored in a directory with subfolders `0` (IDC negative) and `1` (IDC positive).

```python
dataset_dir = "path/to/breast_cancer_dataset"  # Update with actual path
image_size = (50, 50)  # Resize images for consistency
features = []
labels = []

# Function to extract HOG features
def extract_hog_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, image_size)
    hog_features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    return hog_features

# Load images and labels
for label in ['0', '1']:
    folder_path = os.path.join(dataset_dir, label)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        if img_path.endswith('.png'):
            hog_features = extract_hog_features(img_path)
            features.append(hog_features)
            # Map IDC labels to priority: 1 (positive) = High, 0 (negative) = Low
            labels.append('High' if label == '1' else 'Low')

# Convert to arrays
X = np.array(features)
y = np.array(labels)

# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
```

## Train Random Forest Model
Train a Random Forest classifier with hyperparameter tuning.

```python
# Define model and parameter grid
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

# Perform grid search
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_rf = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)
```

## Evaluate Model
Evaluate the model on the test set using accuracy and F1-score.

```python
# Predict on test set
y_pred = best_rf.predict(X_test)

# Compute metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score (Macro): {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=['Low', 'High'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
```

## Save Results
Save performance metrics to a file.

```python
metrics = {
    'Accuracy': accuracy,
    'F1-Score (Macro)': f1
}
pd.DataFrame([metrics]).to_csv('performance_metrics.csv', index=False)
print("Metrics saved to performance_metrics.csv")
```