# Iris Dataset Logistic Regression Model

## Project Description

This project uses the **Iris Dataset** and applies **Binary Logistic Regression** to predict species classification based on different features of the flowers (like petal length, petal width, sepal length, and sepal width). The goal is to build a machine learning model using scikit-learn to predict the class of the iris species.

### Objective:
- Predict whether the iris flower belongs to one of two possible species using binary logistic regression.
- The dataset is divided into training and testing sets, and the performance of the model is evaluated based on accuracy.

## Libraries Used

- `numpy`: A fundamental package for numerical computing.
- `pandas`: A library used for data manipulation and analysis.
- `sklearn.model_selection.train_test_split`: To split the dataset into training and testing sets.
- `sklearn.linear_model.LogisticRegression`: To create the logistic regression model.
- `sklearn.datasets.load_iris`: To load the Iris dataset.
- `sklearn.metrics.accuracy_score`: To evaluate the accuracy of the model.

## Installation

To run this project, you need to install the required libraries. You can install them using `pip`:

```bash
pip install numpy pandas scikit-learn
