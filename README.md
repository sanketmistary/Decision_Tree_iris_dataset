
# Decision Tree for Iris Classification

This repository contains a Jupyter Notebook that demonstrates the process of building and evaluating a Decision Tree Classifier for the classic Iris dataset.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Notebook Steps](#notebook-steps)
- [Libraries Used](#libraries-used)
- [Results](#results)

## Introduction
This notebook provides a step-by-step guide to classifying Iris species using a Decision Tree algorithm. It covers data loading, exploratory data analysis, data preprocessing, model training, visualization of the decision tree, and comprehensive model evaluation.

## Dataset
The project uses the **Iris dataset**, a well-known dataset in machine learning and statistics. It contains 150 samples of Iris flowers, each with four features (sepal length, sepal width, petal length, and petal width) and their corresponding species (setosa, versicolor, or virginica).

## Notebook Steps
1.  **Load and Explore Data**: The Iris dataset is loaded, and its basic properties (first 5 rows, info, class distribution) are displayed.
2.  **Data Visualization**: Visualizations like box plots and scatter plots are used to understand the distribution of features and relationships between them across different species.
3.  **Preprocessing**: The data is split into features (X) and target (y), and then further divided into training and testing sets (80% train, 20% test).
4.  **Train Decision Tree Model**: A `DecisionTreeClassifier` is initialized with `max_depth=3` and trained on the preprocessed data.
5.  **Visualize the Decision Tree**: The trained decision tree is visualized to understand its decision-making process.
6.  **Model Evaluation**: The model's performance is assessed using various metrics:
    -   Accuracy Score
    -   Confusion Matrix (visualized with a heatmap)
    -   Classification Report (precision, recall, f1-score)
    -   ROC AUC Score (for multi-class classification)
    -   ROC Curves (One-vs-Rest for each species)
7.  **Feature Importance**: The importance of each feature in the decision-making process is calculated and visualized.

## Libraries Used
-   `pandas`: For data manipulation and analysis.
-   `numpy`: For numerical operations.
-   `matplotlib`: For creating static, interactive, and animated visualizations.
-   `seaborn`: For high-level statistical data visualization.
-   `scikit-learn`: For machine learning tools, including dataset loading, model selection, decision tree algorithms, and metrics.

## Results
The Decision Tree Classifier achieved high accuracy on the Iris dataset, effectively classifying the three species. The visualizations provide insights into feature distributions, model decisions, and feature importance.
ing or generate with AI.
Colab paid products - Cancel contracts here
