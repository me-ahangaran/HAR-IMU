# HAR Model Prediction Power & Correlation Analysis

This folder contains visualizations related to the performance of the Human Activity Recognition (HAR) model and correlation analysis between IMU devices. It includes **activity prediction plots** and **heatmaps** that compare the ground truth and predicted labels at different timestamps, as well as **correlation visualizations** for different IMU device scores across various placements (hand, chest, and ankle).

---

## Activity Prediction Plots

The activity prediction plots compare the ground truth (dashed line) and predicted (solid line) labels at different timestamps. These plots show the prediction power of the HAR model by visualizing the predicted and actual activity labels over time for different IMU devices. The following steps were followed to generate the plots:

1. **Data Averaging**: 
   - IMU sensor parameters were averaged over time for each activity and for each device placement (hand, chest, ankle).
   - This helps in visualizing the overall prediction power of the model by reducing the noise inherent in the raw sensor data.

2. **Prediction vs Ground Truth**:
   - Each plot presents the comparison between the **predicted labels** (solid line) and **ground truth labels** (dashed line).
   - This enables an understanding of how accurately the model is able to predict the activities based on the sensor data from different body placements.

3. **Devices & Placements**:
   - The plots represent data from three IMU devices placed on:
     - **Hand**
     - **Chest**
     - **Ankle**
   
4. **Sample Activity Plots**
<image src = "https://github.com/me-ahangaran/HAR-IMU/Plots/Activity plots/subject108_Ankle_Acc_3D_16g.svg">
<image src = "https://github.com/me-ahangaran/HAR-IMU/Plots/Activity plots/subject108_Ankle_Gyroscope_3D.svg">
<image src = "https://github.com/me-ahangaran/HAR-IMU/blob/main/Plots/Activity%20plots/subject108_Ankle_Magnetometer_3D.svg">
   
The prediction accuracy for each activity is highlighted by comparing the two lines. These plots help assess how well the model generalizes across different sensor placements.

---

## Correlation Heatmaps

The heatmap diagrams provide a visual representation of the correlation between the scores (feature values) of different IMU devices at different placements. These heatmaps show the degree of relationship between the features and their interactions, helping to understand the underlying structure of the data.

1. **Heatmap Details**:
   - Each heatmap represents correlations between different features collected by IMU devices located at various body placements (hand, chest, ankle).
   - The correlation coefficient is displayed on the heatmap, where **1** represents perfect correlation, **0** represents no correlation, and **-1** represents perfect negative correlation.

2. **Insights from the Heatmaps**:
   - The heatmaps help identify which IMU features from different body placements share strong correlations, which can be useful for understanding feature redundancy and selecting the most informative features for model training.
   - The correlations between device placements show how similar or distinct the features are when collected from different parts of the body, which can influence the model's performance.

---

## Example Files

The folder contains the following types of plots:
1. **Activity Prediction Plots**: Visual representations showing the comparison between ground truth and predicted activities.
2. **Correlation Heatmaps**: Diagrams visualizing the relationships between the features of different IMU devices at various placements.

These visualizations help in assessing the effectiveness of the HAR model and in providing insights into how sensor data from different body placements influences activity recognition.

---

## How to Use These Visualizations

To understand the model's performance and feature relationships, you can:
1. Review the **activity prediction plots** to evaluate how well the model predicts each activity.
2. Analyze the **correlation heatmaps** to see how sensor features correlate with each other and how different placements impact feature relationships.

---

## Conclusion

The visualizations in this folder play a crucial role in understanding the prediction power of the HAR model and the relationships between sensor data from various IMU devices. They provide insights into both model performance and feature interactions, helping to refine and optimize the system.
