## train Random Forest classifier for HAR task using IMU data and generate activity plots

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Constants
TRAIN_SUBJECTS = [f'./Protocol/subject10{i}.csv' for i in range(1, 8)]
TEST_SUBJECTS = [f'./Protocol/subject10{i}.csv' for i in range(8, 9)]
DATA_PATH = './'
MODEL_PATH = './trained_model_RF/'
OUTPUT_PATH = './Output_RF.txt'
PLOTS_PATH = './Plots/'
ACTIVITY_LABELS = {
    1: 'lying', 2: 'sitting', 3: 'standing', 4: 'walking', 5: 'running', 6: 'cycling', 7: 'Nordic walking',
    9: 'watching TV', 10: 'computer work', 11: 'car driving', 12: 'ascending stairs', 13: 'descending stairs',
    16: 'vacuum cleaning', 17: 'ironing', 18: 'folding laundry', 19: 'house cleaning', 20: 'playing soccer', 24: 'rope jumping'
}

# Ensure output directories exist
os.makedirs(PLOTS_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# Function to plot activity intervals
def plot_activity_intervals(ax, times, activities, linestyle, offset):
    for activity in np.unique(activities):
        mask = (activities == activity)
        start_indices = np.where(np.diff(mask.astype(int)) == 1)[0] + 1
        end_indices = np.where(np.diff(mask.astype(int)) == -1)[0] + 1
        if mask[0]:
            start_indices = np.insert(start_indices, 0, 0)
        if mask[-1]:
            end_indices = np.append(end_indices, len(activities))
        for start, end in zip(start_indices, end_indices):
            ax.hlines(y=activity + offset, xmin=times[start], xmax=times[end-1],
                      colors=activity_colors[activity],
                      linestyles=linestyle, linewidth=2)

# Load training data
X_train = np.load('X_train(101-107).npy')
y_train = np.load('y_train(101-107).npy')

# Train model
# print("Training RandomForestClassifier...")
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# Save model
model_file = os.path.join(MODEL_PATH, 'rf_har_model.pkl')
# joblib.dump(model, model_file)
# print(f"Model saved at {model_file}")

# Load model for testing
model = joblib.load(model_file)

# Testing and plotting
output_lines = []
print("Starting testing phase...")
for file in TEST_SUBJECTS:
    print(f"Testing on {file}...")
    df = pd.read_csv(os.path.join(DATA_PATH, file), header=None)
    df = df[df[1] != 0]  # Remove activity ID = 0 rows
    # imputer = KNNImputer(n_neighbors=5)
    # df.iloc[:, 3:] = imputer.fit_transform(df.iloc[:, 3:])
    # scaler = StandardScaler()
    # df.iloc[:, 3:] = scaler.fit_transform(df.iloc[:, 3:])
    # X_test = df.iloc[:, 3:].values
    # y_test = df.iloc[:, 1].values
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

    # Feature subsets
    feature_subsets = {
        'Temperature': [0],
        'Acc_3D_16g': [1, 2, 3],
        'Acc_3D_6g': [4, 5, 6],
        'Gyroscope_3D': [7, 8, 9],
        'Magnetometer_3D': [10, 11, 12],
        'Orientation': [13, 14, 15, 16]
    }
    device_starts = {'Hand': 3, 'Chest': 20, 'Ankle': 37}

    # Assign colors
    activity_colors = {activity: plt.cm.tab20(i % 20) for i, activity in enumerate(ACTIVITY_LABELS.keys())}

    # Plotting
    times = df[0].values
    for device, start_col in device_starts.items():
        for subset_name, indices in feature_subsets.items():
            cols = [start_col + idx for idx in indices]
            subset_data = df.iloc[:, cols].mean(axis=1)

            fig, ax = plt.subplots(figsize=(20, 6))
            ax.plot(times, subset_data, label=f'{device} {subset_name} Mean', color='darkgrey')

            # Plot GT and predicted lines at higher Y positions
            offset_gt = 4
            offset_pred = 8
            plot_activity_intervals(ax, times, y_test, linestyle='dashed', offset=offset_gt)
            plot_activity_intervals(ax, times, y_pred, linestyle='solid', offset=offset_pred)

            # Legend construction
            legend_elements = [
                Line2D([0], [0], color='black', linestyle='dashed', label='Ground Truth'),
                Line2D([0], [0], color='black', linestyle='solid', label='Predicted')
            ]
            # One solid line per activity
            for activity in sorted(np.unique(np.concatenate([y_test, y_pred]))):
                legend_elements.append(Line2D([0], [0], color=activity_colors[activity],
                                              linestyle='solid', label=f'{ACTIVITY_LABELS[activity]}'))

            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
            ax.set_title(f'{device} IMU Device - {subset_name} Features')
            ax.set_xlabel('Time')
            ax.set_ylabel('Feature Mean / Activity ID')

            # Adjust Y-limits so GT and predicted lines are above
            min_y = min(subset_data.min(), min(y_test)) - 1
            max_y = max(subset_data.max(), max(y_test)) + 5
            ax.set_ylim(min_y, max_y)

            # Save plot
            filename = f'{file.split("/")[-1].split(".")[0]}_{device}_{subset_name}.svg'
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_PATH, filename), format='svg')
            plt.close()
            print(f"Plot saved: {filename}")

# Write output metrics
with open(OUTPUT_PATH, 'w') as f:
    f.writelines(output_lines)
print(f"Results saved to {OUTPUT_PATH}")
