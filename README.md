# Human Activity Recognition (HAR) using Inertial Measurement Unit‎ (IMU) sensory data

This repository contains the implementation of a **Random Forest-based Human Activity Recognition (HAR)** system using the **PAMAP2 Physical Activity Monitoring dataset**. The system processes IMU sensor data collected from different body positions to predict human activities and provides performance visualizations and evaluations.

---

## Dataset Information

The PAMAP2 dataset consists of synchronized IMU sensor data and heart rate measurements collected from multiple subjects performing daily and sports activities.

### Data Collection Setup

- **Sensors Used**:
  - **3 Colibri Wireless IMUs**:
    - Sampling Frequency: 100Hz
    - Positions:
      - Wrist (dominant arm)
      - Chest
      - Ankle (dominant side)
    - Sensors:
      - 3D-Acceleration
      - 3D-Gyroscope
      - 3D-Magnetometer
      - Orientation (invalid in dataset)
  - **Heart Rate Monitor**:
    - BM-CS5SR from BM Innovations GmbH
    - Sampling Frequency: ~9Hz
  - **Companion Device**:
    - Viliv S5 UMPC (Intel Atom Z520 CPU, 1.33GHz, 1GB RAM)
    - Used for labeling activities via a GUI

### Subjects

- **Total Participants**: 9
  - **Gender**: 8 males, 1 female
  - **Age**: 27.22 ± 3.31 years
  - **BMI**: 25.11 ± 2.62 kg/m²
- All participants provided consent for scientific use of data.

### Activities

Each participant performed a set of 12 predefined activities, with some additional optional activities. Below are the key activities used:

| Activity ID | Activity Name         |
|------------:|----------------------|
| 1           | Lying                 |
| 2           | Sitting               |
| 3           | Standing              |
| 4           | Walking               |
| 5           | Running               |
| 6           | Cycling               |
| 7           | Nordic Walking        |
| 12          | Ascending Stairs      |
| 13          | Descending Stairs     |
| 16          | Vacuum Cleaning       |
| 17          | Ironing               |
| 24          | Rope Jumping          |

---

## Project Workflow

1. **Preprocessing**:
   - Handle missing IMU sensor data using **KNN imputation**.
   - Normalize IMU feature values.
   - Extract and organize 51 features: 17 from each IMU (wrist, chest, ankle).
  
2. **Model Training**:
   - **Random Forest Classifier** is used.
   - Training conducted on data from 7 participants.
   - Testing performed on data from the 8th participant.

3. **Evaluation & Visualization**:
   - Classification reports generated (accuracy, precision, recall, F1-score).
   - Heatmaps created to visualize confusion matrix.
   - IMU feature mean plots over time.
   - Activity prediction plots showing both ground truth and predicted activities.

---

## Evaluation Results

- **Total Accuracy**: `0.5624`
  
| Activity             | F1-Score |
|---------------------:|:--------:|
| Lying                | 0.97     |
| Sitting              | 0.81     |
| Walking              | 0.98     |
| Cycling              | 0.79     |
| Running              | 0.48     |
| Ascending Stairs     | 0.37     |
| Descending Stairs    | 0.38     |
| Vacuum Cleaning      | 0.58     |
| Standing             | 0.01     |
| Nordic Walking       | 0.11     |
| Ironing              | 0.02     |
| Rope Jumping         | 0.00     |

---

## Visualization Samples

The repository includes:

- **Activity Prediction Plots**: Ground truth and predicted activities displayed with distinct lines.
- **Confusion Matrix Heatmap**: Easy-to-understand visualization of classification accuracy.
- **IMU Feature Plots**: Averaged IMU feature trends per activity over time.

---

## Package Requirements

To run the project, ensure you have the following Python packages installed:

```bash
numpy
pandas
scikit-learn
seaborn
matplotlib
scipy
```

You can install them via:

```bash
pip install numpy pandas scikit-learn seaborn matplotlib scipy
```

---

## References

1. A. Reiss and D. Stricker. Introducing a New Benchmarked Dataset for Activity Monitoring. The 16th IEEE International Symposium on Wearable Computers (ISWC), 2012.
2. A. Reiss and D. Stricker. Creating and Benchmarking a New Dataset for Physical Activity Monitoring. The 5th Workshop on Affect and Behaviour Related Assistance (ABRA), 2012.

