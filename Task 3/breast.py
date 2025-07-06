# Step 1: Import Libraries
"""
This script performs automated diagnosis of breast cancer using HOG (Histogram of Oriented Gradients) features and a Random Forest classifier. The workflow includes:
1. Importing necessary libraries for image processing, feature extraction, machine learning, and visualization.
2. Preprocessing the dataset:
    - Extracts HOG features from grayscale images in the training set.
    - Saves or loads extracted features to/from a CSV file for efficiency.
    - Assigns labels ('Low' for benign, 'High' for malignant).
3. Splitting the dataset into training, validation, and test sets (70/15/15 split) with stratification.
4. Training a Random Forest classifier using grid search for hyperparameter optimization, focusing on macro F1-score.
5. Evaluating the best model on the validation and test sets, reporting accuracy, F1-score, and a detailed classification report.
6. Visualizing the confusion matrix for the test set.
7. Saving performance metrics to a CSV file.
Configuration options include dataset directory, image size, and feature CSV filename. The script is designed for environments with limited memory and CPU resources.
"""
# This code is designed to run in a Python environment with the necessary libraries installed.
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

# Step 2: Preprocess Data
# Configuration
dataset_dir = r"C:\Users\Tmseb\OneDrive\Desktop\iuss-23-24-automatic-diagnosis-breast-cancer\complete_set"
image_size = (50, 50)  # Small size for low memory
feature_csv = "hog_features.csv"

# Function to extract HOG features
def extract_hog_features(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Failed to load {image_path}")
            return None
        image = cv2.resize(image, image_size)
        hog_features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        return hog_features
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Check if feature CSV exists
if Path(feature_csv).exists():
    print(f"Loading features from {feature_csv}")
    df = pd.read_csv(feature_csv)
    X = df.drop(columns=['label']).to_numpy()
    y = df['label'].to_numpy()
else:
    print("Extracting features from training_set...")
    features = []
    labels = []
    total_images = 0
    processed_images = 0

    # Count total images
    for category in ['benign', 'malignant']:
        folder_path = os.path.join(dataset_dir, 'training_set', category)
        total_images += len([f for f in os.listdir(folder_path) if f.endswith('.png') and '_mask' not in f])

    # Process images
    for category, priority in [('benign', 'Low'), ('malignant', 'High')]:
        folder_path = os.path.join(dataset_dir, 'training_set', category)
        for img_name in os.listdir(folder_path):
            if img_name.endswith('.png') and '_mask' not in img_name:
                img_path = os.path.join(folder_path, img_name)
                hog_features = extract_hog_features(img_path)
                if hog_features is not None:
                    features.append(hog_features)
                    labels.append(priority)
                    processed_images += 1
                    if processed_images % 100 == 0:
                        print(f"Processed {processed_images}/{total_images} images")

    # Save to CSV
    feature_names = [f'hog_{i}' for i in range(len(features[0]))]
    df = pd.DataFrame(features, columns=feature_names)
    df['label'] = labels
    df.to_csv(feature_csv, index=False)
    print(f"Features saved to {feature_csv}")
    X = np.array(features)
    y = np.array(labels)

# Split dataset (70% train, 15% validation, 15% test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Step 3: Train Random Forest Model
# Define Random Forest model
rf = RandomForestClassifier(random_state=42)

# Lightweight parameter grid for limited resources
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

# Perform grid search
print("Training Random Forest with grid search...")
grid_search = GridSearchCV(
    rf,
    param_grid,
    cv=3,  # Reduced folds for faster processing
    scoring='f1_macro',  # Optimize for macro F1-score
    n_jobs=1,  # Single job to avoid CPU overload
    verbose=1
)
grid_search.fit(X_train, y_train)

# Best model
best_rf = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation F1-score:", grid_search.best_score_)

# Evaluate on validation set
y_val_pred = best_rf.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred, average='macro')
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation F1-Score (Macro): {val_f1:.4f}")

# Step 4: Evaluate Model on Test Set
print("\nEvaluating model on test set...")
y_test_pred = best_rf.predict(X_test)

# Compute metrics
test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred, average='macro')

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test F1-Score (Macro): {test_f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_test_pred, labels=['Low', 'High'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
plt.title('Confusion Matrix (Test Set)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Save performance metrics
metrics = {
    'Accuracy': test_accuracy,
    'F1-Score (Macro)': test_f1
}
pd.DataFrame([metrics]).to_csv('performance_metrics.csv', index=False)
print("Metrics saved to performance_metrics.csv")