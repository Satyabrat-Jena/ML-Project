# Weather Prediction Project

## Overview
This project aims to predict future temperatures based on historical weather data using various machine learning models. The project involves data preprocessing, feature engineering, model training, evaluation, and hyperparameter tuning to achieve accurate predictions.

## Data
The dataset used in this project contains historical weather data for Bhubaneswar, including monthly temperatures from 2000 to 2021. The data is stored in a CSV file named `Bhubaneswar.csv`.

## Methodology
1. **Data Preprocessing**:
   - Convert the 'Date' column to datetime format.
   - Create dummy variables for the 'Month' column.
   - Scale the features for better model performance.

2. **Model Selection and Training**:
   - Train multiple machine learning models including Linear Regression, Random Forest, and Gradient Boosting Regressors.
   - Evaluate the models based on R-squared and Mean Squared Error (MSE) metrics.

3. **Hyperparameter Tuning**:
   - Use GridSearchCV to find the best hyperparameters for the Random Forest model.
   - Perform cross-validation to ensure robust model performance.

4. **Prediction**:
   - Predict temperatures for the year 2022 using the trained model.
   - Store the predictions in a DataFrame for analysis.

## Results
The project identifies the best-performing model and uses it to predict future temperatures. The results are evaluated based on standard metrics and cross-validation scores to ensure accuracy.

## Running the Code
To run the code, follow these steps:

1. **Install Required Libraries**:
   Ensure you have the following Python libraries installed:
   ```sh
   pip install pandas scikit-learn matplotlib plotly
