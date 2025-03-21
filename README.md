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


```
# in case if you dont remember, here are important facts about the code

```python 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

``` 

### Explanation:

 - `train_test_split()` : This is a function from scikit-learn (imported as from sklearn.model_selection import train_test_split). It is used to split your dataset into two parts:
Training data: The portion of the data that will be used to train the model.
Testing data: The portion of the data that will be used to test the model's performance after training.

This function ensures that the model is trained on one subset of the data and tested on another subset that the model hasn't seen before. This helps evaluate how well the model generalizes to unseen data.

x and y:
 x represents the features (independent variables) of your dataset. In the case of the Student Exam Score Prediction project, x would be the Study Hours.
 y represents the target (dependent variable), which in this case would be the Exam Scores.

`X_train, X_test, y_train, y_test`: These are the four variables that will hold the split data:
 `X_train`: The training set features (study hours).
 `X_test`: The test set features (study hours).
 `y_train`: The training set target (exam scores).
 `y_test`: The test set target (exam scores).

`test_size=0.2` : This parameter controls the proportion of the dataset that will be used for testing. In this case, test_size=0.2 means that 20% of the data will be used for testing, and the remaining 80% will be used for training.
You can adjust this ratio based on how much data you want to allocate to the training or testing sets.

`random_state=42`: This parameter ensures reproducibility. The random_state is a seed for the random number generator used when splitting the data. By specifying a value (in this case, 42), you ensure that every time you run the code, the data will be split the same way, making the results consistent. If you don't set random_state, the split may vary each time you run the code.
You can use any number (42 is just a commonly used "seed" value).

## What Happens:

The data (x and y) is randomly shuffled and split into two sets.
80% of the data will be used to train the model (X_train and y_train), and 20% will be used to test the model's performance (X_test and y_test).

This way, you train the model on one set of data and evaluate it on a separate set, preventing overfitting and giving a better estimate of how the model will perform on new, unseen data. 

## virtual environment
A virtual environment in VS Code (or in general) is an isolated workspace where you can install Python packages without affecting the global Python environment on your system. This is useful for keeping dependencies organized and avoiding conflicts between different projects
