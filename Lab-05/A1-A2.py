import pandas as pd
import numpy as np
import logging
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Logging Configuration
logging.basicConfig(level=logging.INFO)

def load_data(file_path):
    """Loads dataset from a CSV file."""
    return pd.read_csv(file_path)

def split_features_target(data, x_column, y_column):
    """Splits dataset into features (X) and target (Y)."""
    x = data[[x_column]]
    y = data[y_column]
    return x, y

def split_train_test(x, y, test_size=0.2, random_state=43):
    """Splits the dataset into training and testing sets."""
    return train_test_split(x, y, test_size=test_size, random_state=random_state)

def remove_outliers(x_train, y_train, threshold=3):
    """Removes outliers from training data using Z-score method."""
    z_scores = np.abs(zscore(x_train))
    mask = (z_scores < threshold).all(axis=1)
    return x_train[mask], y_train[mask]

def train_model(x_train, y_train):
    """Trains a Linear Regression model."""
    model = LinearRegression().fit(x_train, y_train)
    return model

def make_predictions(model, x):
    """Generates predictions using the trained model."""
    return model.predict(x)

def evaluate_model(y_true, y_pred):
    """Evaluates the model using various metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100  # Avoid division by zero
    r2 = r2_score(y_true, y_pred)
    return {
        "Mean Squared Error": mse,
        "Root Mean Squared Error": rmse,
        "Mean Absolute Percentage Error": mape,
        "R2 Score": r2
    }

if __name__ == "__main__":
    file_path = "GST - C_AST.csv"
    data = load_data(file_path)
    
    x, y = split_features_target(data, 'ast_embedding_0', 'Final_Marks')
    x_train, x_test, y_train, y_test = split_train_test(x, y)
    logging.info("Data Split into training and testing data\n")
    
    x_train_cleaned, y_train_cleaned = remove_outliers(x_train, y_train)
    logging.info(f"Removed {len(x_train) - len(x_train_cleaned)} outliers from training data\n")
    
    model = train_model(x_train_cleaned, y_train_cleaned)
    logging.info("Model Created and Fitted\n")
    
    y_train_pred = make_predictions(model, x_train_cleaned)
    y_test_pred = make_predictions(model, x_test)
    logging.info("Predictions done for training and testing data\n")
    
    train_metrics = evaluate_model(y_train_cleaned, y_train_pred)
    test_metrics = evaluate_model(y_test, y_test_pred)
    
    logging.info("Model Evaluation Completed\n")


    print("Training Metrics:")
    for metric, value in train_metrics.items():
        print(f"{metric}: {value}")
    print()
    print("Testing Metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value}")
