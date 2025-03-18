## train Random Forest classifier for HAR task using IMU data and generate heat map plots

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Constants
TRAIN_SUBJECTS = [f'./Protocol/subject10{i}.csv' for i in range(1, 3)] # should be (1, 8)
TEST_SUBJECTS = [f'./Protocol/subject10{i}.csv' for i in range(8, 9)] # should be (8, 10)
DATA_PATH = './'
MODEL_PATH = './trained_model_RF/'
OUTPUT_PATH = './Output_RF.txt'
ACTIVITY_LABELS = {
    1: 'lying', 2: 'sitting', 3: 'standing', 4: 'walking', 5: 'running', 6: 'cycling', 7: 'Nordic walking',
    9: 'watching TV', 10: 'computer work', 11: 'car driving', 12: 'ascending stairs', 13: 'descending stairs',
    16: 'vacuum cleaning', 17: 'ironing', 18: 'folding laundry', 19: 'house cleaning', 20: 'playing soccer', 24: 'rope jumping'
}

# Function to load and preprocess data
def load_and_preprocess(files):
    all_data = []
    print("Loading and preprocessing data...")
    for file in files:
        print(f"Processing {file}...")
        df = pd.read_csv(os.path.join(DATA_PATH, file), header=None)
        df = df[df[1] != 0]  # Remove activity ID = 0 rows
        # Impute missing values
        imputer = KNNImputer(n_neighbors=5)
        df.iloc[:, 3:] = imputer.fit_transform(df.iloc[:, 3:])
        # Normalize features
        scaler = StandardScaler()
        df.iloc[:, 3:] = scaler.fit_transform(df.iloc[:, 3:])
        all_data.append(df)
    combined = pd.concat(all_data, ignore_index=True)
    X = combined.iloc[:, 3:].values
    y = combined.iloc[:, 1].values
    return X, y

# Load training data
# X_train, y_train = load_and_preprocess(TRAIN_SUBJECTS)
X_train = np.load('X_train(101-107).npy')
y_train = np.load('y_train(101-107).npy')

# Train model
# print("Training RandomForestClassifier...")
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# Save model
os.makedirs(MODEL_PATH, exist_ok=True)
model_file = os.path.join(MODEL_PATH, 'rf_har_model.pkl')
# joblib.dump(model, model_file)
# print(f"Model saved at {model_file}")

# Load model for testing
model = joblib.load(model_file)

# Testing and evaluation
output_lines = []
print("Starting testing phase...")
for file in TEST_SUBJECTS:
    print(f"Testing on {file}...")
    df = pd.read_csv(os.path.join(DATA_PATH, file), header=None)
    df = df[df[1] != 0]  # Remove activity ID = 0 rows
    imputer = KNNImputer(n_neighbors=5)
    df.iloc[:, 3:] = imputer.fit_transform(df.iloc[:, 3:])
    scaler = StandardScaler()
    df.iloc[:, 3:] = scaler.fit_transform(df.iloc[:, 3:])
    X_test = df.iloc[:, 3:].values
    y_test = df.iloc[:, 1].values
    # np.save(f'{file[:-4]}_X_test.npy', X_test)
    # np.save(f'{file[:-4]}_y_test.npy', y_test)
    X_test = np.load(f'{file[:-4]}_X_test.npy')
    y_test = np.load(f'{file[:-4]}_y_test.npy')
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=[ACTIVITY_LABELS[i] for i in sorted(ACTIVITY_LABELS.keys()) if i in y_test])
    output_lines.append(f"Results for {file}:\n")
    output_lines.append(f"Accuracy: {acc:.4f}\n")
    output_lines.append(f"Classification Report:\n{report}\n")
    output_lines.append(f"Confusion Matrix:\n{cm}\n\n")

    # Plot original IMU signals and predicted activities
    plt.figure(figsize=(20, 6))
    plt.plot(df[0], y_test, label='Ground Truth', alpha=0.7)
    plt.plot(df[0], y_pred, label='Predicted', alpha=0.7)
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Activity ID')
    plt.title(f'Activity Prediction for {file}')
    plt.legend()
    plt.savefig(f'{file[:-4]}_activity_plot_RF.svg', format='SVG', dpi=300, bbox_inches='tight')
    plt.close()

# Correlation analysis
print("Calculating correlation matrices...")
def plot_correlation(subset_name, data, cols):
    corr = pd.DataFrame(data[:, cols]).corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.title(f'Correlation Matrix - {subset_name}')
    plt.savefig(f'correlation_RF_{subset_name}.svg', format='SVG', dpi=300, bbox_inches='tight')
    plt.close()

# Subsets indices
subset_indices = {
    'Temperature': [0],
    'Acc_3D_16g': [1, 2, 3],
    'Acc_3D_6g': [4, 5, 6],
    'Gyroscope_3D': [7, 8, 9],
    'Magnetometer_3D': [10, 11, 12],
    'Orientation': [13, 14, 15, 16]
}

for device, start_col in zip(['Hand', 'Chest', 'Ankle'], [0, 17, 34]):
    for subset, indices in subset_indices.items():
        cols = [start_col + i for i in indices]
        plot_correlation(f'{device}_{subset}', X_train, cols)

# Save output
with open(OUTPUT_PATH, 'w') as f:
    for line in output_lines:
        f.write(line)
print(f"All outputs saved to {OUTPUT_PATH}")
print("Process completed.")
