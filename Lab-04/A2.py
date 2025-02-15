import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def load_data(file_path, sheet_name):
    # Load dataset from an Excel file.
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    return data

def segregate_data(data):
    # Segregate features (A) and target variable (C) from the dataset.
    A = np.array(data[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]])  # Features
    C = np.array(data[["Payment (Rs)"]])  # Target variable (total payment)
    return A, C

def calculate_pseudo_inverse(A):
    # Compute the pseudo-inverse of matrix A using NumPy's pinv function.
    return np.linalg.pinv(A)

def calculate_model_vector(A_pseudo_inv, C):
    # Compute the model vector X using the pseudo-inverse approach.
    return np.dot(A_pseudo_inv, C)

def calculate_predictions(A, X):
    # Generate predicted payment values using the model.
    return np.dot(A, X)

def evaluate_model(y_true, y_pred):
    # Compute evaluation metrics: MSE, RMSE, MAPE, and R² Score.
    mse = mean_squared_error(y_true, y_pred)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Mean Absolute Percentage Error
    r2 = r2_score(y_true, y_pred)  # R² Score (coefficient of determination)
    return mse, rmse, mape, r2

if __name__ == "__main__":
    # Define file path and sheet name
    file_path = "lab_session_data.xlsx"
    sheet_name = 'Purchase data'
    
    # Load dataset from the Excel file
    data = load_data(file_path, sheet_name)
    
    # Segregate data into features (A) and target variable (C)
    A, C = segregate_data(data)
    
    # Compute the pseudo-inverse of A
    A_pseudo_inv = calculate_pseudo_inverse(A)
    
    # Calculate model vector X
    X = calculate_model_vector(A_pseudo_inv, C)
    
    # Generate predictions for payment values
    C_pred = calculate_predictions(A, X)
    
    # Evaluate model performance using error metrics
    mse, rmse, mape, r2 = evaluate_model(C, C_pred)
    
    # Print evaluation results
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"R² Score: {r2:.4f}")
