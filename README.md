# Student Exam Score Prediction using Linear Regression

This project demonstrates how to predict student exam scores based on the number of hours they study using a simple **Linear Regression** model. It utilizes Python's **scikit-learn** library for implementing the machine learning model and **matplotlib** for data visualization.

## Project Overview

The goal of this project is to explore the relationship between the number of study hours and the exam scores achieved by students. By training a linear regression model, we can predict how much a student might score based on the time they dedicate to studying.

## Key Features

- **Dataset**: A small, manually created dataset with study hours and corresponding exam scores.
- **Data Preprocessing**: Simple data splitting into training and testing sets.
- **Model Training**: Trains a linear regression model using the scikit-learn library.
- **Evaluation Metrics**: Uses **Mean Squared Error (MSE)** and **R² Score** to evaluate the performance of the model.
- **Visualization**: Plots a graph to visualize the relationship between study hours and exam scores, as well as the fitted regression line.

## Technologies Used

- **Python**
- **scikit-learn** (for machine learning)
- **pandas** (for data manipulation)
- **matplotlib** (for data visualization)

## How It Works

1. A dataset of study hours and exam scores is created.
2. The data is split into **features** (study hours) and the **target variable** (exam scores).
3. The data is further split into **training** and **testing** sets to evaluate the model's performance.
4. A **linear regression model** is trained on the training data.
5. The model is evaluated using testing data and metrics like **MSE** and **R²**.
6. Finally, a graph is plotted to visualize the model’s performance.

## Requirements

Make sure you have the following Python libraries installed:

- `scikit-learn`
- `pandas`
- `matplotlib`

You can install them using pip:

```bash
pip install scikit-learn pandas matplotlib
