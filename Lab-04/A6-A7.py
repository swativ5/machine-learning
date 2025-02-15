import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import logging
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


def load_data(file_path):
    """Loads the dataset from an Excel file into a Pandas DataFrame."""
    dataset = pd.read_excel(file_path)
    return pd.DataFrame(dataset)

def clean_data(df):
    """Removes unnecessary columns and converts the target variable into a binary format."""
    columns_to_remove = ['Question', 'Comprehensible_Code_with_logical_errors', 'Comprehensible_code_with_syntax_errors',
                           'Correct_code_and_output', 'Correct_Code', 'Code_with_Error', 'code_processed',
                           'code_with_question', 'code_comment', 'code_with_solution', 'ast', 'Final_Marks', 'Incomprehensible_Code']
    df.drop(columns=columns_to_remove, inplace=True)

    # Convert 'Header_and_Main_Declaration' into a binary class (0 or 1)
    df["Header_and_Main_Declaration"] = df["Header_and_Main_Declaration"].apply(lambda x: 1 if x > 1 else 0)
    return df

def process_error_types(df):
    """Identifies unique error types from 'Type_of_Error' and applies one-hot encoding."""
    unique_errors = set()
    for entry in df['Type_of_Error']:
        error_list = entry.strip("[]").replace("'", "").split(", ")  # Cleaning text format
        unique_errors.update(error_list)

    # Apply one-hot encoding for each unique error type
    for error in unique_errors:
        df[error] = df['Type_of_Error'].apply(lambda x: 1 if error in x else 0)

    df.drop(columns=['Type_of_Error'], inplace=True)
    df.drop(columns=['2b', '2d', '1d', '2c', '3a', '1a', '2a', '1e'], inplace=True)
    return df, unique_errors

def prepare_train_test_sets(df):
    """Splits the dataset into training and testing subsets after handling missing values."""
    df.fillna(df.median(), inplace=True)  # Handle missing values
    X = df.drop(columns=['Header_and_Main_Declaration'])  # Extract features
    y = df['Header_and_Main_Declaration']  # Extract labels
    return train_test_split(X, y, test_size=0.3, random_state=42)
   
def train_knn_classifier(X_train, y_train, k=3):
    """Trains a K-Nearest Neighbors (KNN) model with the given k value."""
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    return knn_model

def test_knn_classifier(knn_model, X_test, y_test):
    """Evaluates the KNN classifier using the test dataset."""
    model_accuracy = knn_model.score(X_test, y_test)
    predicted_labels = knn_model.predict(X_test)
    return model_accuracy, predicted_labels


def predict_knn(knn, test_data):
    """Predict class labels for test data using trained KNN."""
    return knn.predict(test_data)

def plot_training_data(train_data, train_labels):
    """Plot training data with colors based on class labels."""
    plt.figure(figsize=(6, 6))
    
    for i in range(len(train_data)):
        color = 'darkblue' if train_labels[i] == 0 else 'darkred'
        plt.scatter(train_data[i, 0], train_data[i, 1], 
                    color=color, edgecolors='black', marker='o', s=100)
    
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Training Data Scatter Plot")
    plt.grid(True)
    plt.show()

def plot_decision_boundary(train_data, train_labels, test_data, test_predictions):
    """Visualize the KNN decision boundary."""
    plt.figure(figsize=(8, 8))
    
    # Plot test data (decision boundary)
    plt.scatter(test_data[:, 0], test_data[:, 1], 
                c=test_predictions, cmap='coolwarm', alpha=0.3, s=5)
    
    # Plot training data (darker colors)
    for i in range(1, len(train_data)):
        color = 'darkblue' if train_labels[i] == 0 else 'darkred'
        plt.scatter(train_data[i, 0], train_data[i, 1], 
                    color=color, edgecolors='black', marker='o', s=100)
    
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("KNN Decision Boundary")
    plt.show()

def knn_decision_boundary(train_data, train_labels, test_data, k_values):
    """Train KNN with different k-values and visualize decision boundaries."""
    fig, axes = plt.subplots(1, len(k_values), figsize=(15, 5))

    for i, k in enumerate(k_values):
        knn = train_knn_classifier(train_data, train_labels, k)
        test_predictions = predict_knn(knn, test_data)

        axes[i].scatter(test_data[:, 0], test_data[:, 1], c=test_predictions, cmap='coolwarm', alpha=0.1)
        axes[i].scatter(train_data[:, 0], train_data[:, 1], c=train_labels, cmap='coolwarm', 
                        edgecolors='black', marker='o', s=100)
        axes[i].set_title(f'k = {k}')
        axes[i].set_xlim(-0.5, 1.5)
        axes[i].set_ylim(-0.5, 1.5)

    plt.show()

def hyperparameter_tuning(train_data, train_labels):
    """Perform hyperparameter tuning using Randomized and Grid Search."""
    # Define the parameter grid
    param_grid = {
        'n_neighbors': [1, 3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    
    # Initialize the KNN classifier
    knn = KNeighborsClassifier()
    
    # Perform Randomized Search
    random_search = RandomizedSearchCV(knn, param_distributions=param_grid, n_iter=10, cv=5)
    random_search.fit(train_data, train_labels)
    
    # Perform Grid Search
    grid_search = GridSearchCV(knn, param_grid=param_grid, cv=5)
    grid_search.fit(train_data, train_labels)
    
    return random_search.best_params_, grid_search.best_params_

if __name__ == "__main__":
    file_path = '15-C.xlsx'

    # Load and preprocess data
    dataset = load_data(file_path)
    dataset = clean_data(dataset)
    dataset, error_types = process_error_types(dataset)
    
    # Prepare training and testing data
    X_train, X_test, y_train, y_test = prepare_train_test_sets(dataset)
    print("Training Data Shape:", X_train.shape, y_train.shape)
    print("Testing Data Shape:", X_test.shape, y_test.shape)
        
    # Train a KNN classifier (k = 3)
    knn_model = train_knn_classifier(X_train, y_train, k=3)
    
    # Evaluate the classifier on test data
    knn_accuracy, knn_predictions = test_knn_classifier(knn_model, X_test, y_test)

    # Plot the training data and decision boundary
    plot_training_data(X_train.to_numpy(), y_train.to_numpy())
    plot_decision_boundary(X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), knn_predictions)

    # Experiment with different k-values
    k_values = [1, 3, 5, 10]
    # Plot decision boundaries for different k-values
    knn_decision_boundary(X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), k_values)

    # Hyperparameter tuning
    random_params, grid_params = hyperparameter_tuning(X_train, y_train)

    randomised_n_neighbors = random_params['n_neighbors']
    grid_search_n_neighbors = grid_params['n_neighbors']

    print(f"Randomized Search Best n_neighbors: {randomised_n_neighbors}")
    print(f"Grid Search Best n_neighbors: {grid_search_n_neighbors}")