import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV



# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def generate_training_data(n_samples=20, seed=42):
    """Generate training data with random points and class labels."""
    np.random.seed(seed)
    train_X = np.random.uniform(1, 10, n_samples)
    train_Y = np.random.uniform(1, 10, n_samples)
    train_labels = np.random.choice([0, 1], size=n_samples)
    
    train_data = np.column_stack((train_X, train_Y))
    return train_data, train_labels

def generate_test_data(x_range=(0, 10.1), y_range=(0, 10.1), step=0.1):
    """Generate a grid of test points."""
    x_values = np.arange(*x_range, step)
    y_values = np.arange(*y_range, step)
    X_test, Y_test = np.meshgrid(x_values, y_values)
    test_data = np.column_stack((X_test.ravel(), Y_test.ravel()))
    
    return test_data


def train_knn_classifier(train_data, train_labels, k=3):
    """Train a KNN classifier on the training data."""
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data, train_labels)
    return knn

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
    for i in range(len(train_data)):
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
        axes[i].set_xlim(0, 10)
        axes[i].set_ylim(0, 10)

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
    # Generate training and test data
    train_data, train_labels = generate_training_data(n_samples=20)
    test_data = generate_test_data()

    # Log data shapes
    logging.info(f"Training Data Shape: {train_data.shape}")
    logging.info(f"Test Data Shape: {test_data.shape}")

    # Train KNN and classify test data
    knn = train_knn_classifier(train_data, train_labels, k=3)
    test_predictions = predict_knn(knn, test_data)

    # Plot results
    plot_training_data(train_data, train_labels)
    plot_decision_boundary(train_data, train_labels, test_data, test_predictions)

    # Experiment with different k-values
    k_values = [1, 3, 5, 10]
    knn_decision_boundary(train_data, train_labels, test_data, k_values)

    # Hyperparameter tuning
    random_params, grid_params = hyperparameter_tuning(train_data, train_labels)
    logging.info(f"Randomized Search Best Parameters: {random_params}")
    logging.info(f"Grid Search Best Parameters: {grid_params}")

    randomised_n_neighbors = random_params['n_neighbors']
    grid_search_n_neighbors = grid_params['n_neighbors']

    # Log hyperparameter tuning results
    logging.info(f"Randomized Search Best n_neighbors: {randomised_n_neighbors}")
    logging.info(f"Grid Search Best n_neighbors: {grid_search_n_neighbors}")