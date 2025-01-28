import numpy as np
import pandas as pd

def load_data(file_path, sheet_name):
    # Load dataset from Excel file.
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    return data

def segregate_data(data):
    # Segregate the data into matrices A and C.
    A = np.matrix(data[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]])
    C = np.matrix(data[["Payment (Rs)"]])
    return A, C

def calculate_dimensions(A, C):
    # Print dimensions of matrices A and C.
    return np.shape(A), np.shape(C)


def calculate_rank(A):
    # Calculate and return the rank of matrix A.
    return np.linalg.matrix_rank(A)

def calculate_pseudo_inverse(A):
    # Calculate the pseudo-inverse of matrix A
    return np.linalg.pinv(A)

def calculate_model_vector(x_pseudo_inv, C):
    # Calculate the model vector X using pseudo-inverse.
    return np.dot(x_pseudo_inv, C)

def classify_customers(data):
    # Classify customers as 'RICH' or 'POOR'.
    classes = []
    for cost in data["Payment (Rs)"]:
        if cost > 200:
            classes.append("RICH")
        else:
            classes.append("POOR")
    return classes

def add_class_to_data(data, classes):
    # Add the 'Class' column to the DataFrame.
    data["Class"] = classes
    return data

if __name__ == "__main__":
    # Load the data
    file_path = "Lab Session Data.xlsx"
    sheet_name = 'Purchase data'
    data = load_data(file_path, sheet_name)
    
    # Segregate data into A (features) and C (target variable)
    A, C = segregate_data(data)

    # Print dimensions of matrices A and C
    dimA, dimC = calculate_dimensions(A, C)
    print(f"Dimensionality  of A: {dimA}")
    print(f"Dimensions of C: {dimC}")

    # Calculate the rank of matrix A
    rank = calculate_rank(A)
    print(f"Rank of matrix A: {rank}")

    # Calculate the pseudo-inverse of A
    pseudo_inv = calculate_pseudo_inverse(A)
    print(f"Pseudo-inverse of A: {pseudo_inv}")

    # Calculate the model vector X using pseudo-inverse
    X = calculate_model_vector(pseudo_inv, C)
    print(f"Model vector X: {X}")

    # Classify customers as 'RICH' or 'POOR'
    classes = classify_customers(data)

    # Add the classification to the data
    data = add_class_to_data(data, classes)

    # Display the final dataframe with class labels
    print(data[["Customer", "Candies (#)", "Mangoes (Kg)", "Milk Packets (#)", "Payment (Rs)", "Class"]])

