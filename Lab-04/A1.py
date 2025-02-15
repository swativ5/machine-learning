import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report


def load_data(file_path):
    """Loads the dataset from an Excel file into a Pandas DataFrame."""
    dataset = pd.read_excel(file_path)
    return pd.DataFrame(dataset)


def clean_data(df):
    """Removes unnecessary columns and converts the target variable into a binary format."""
    columns_to_remove = ['Question', 'Correct_Code', 'Code_with_Error', 'code_processed',
                         'code_with_question', 'code_comment', 'code_with_solution', 'ast']
    df.drop(columns=columns_to_remove, inplace=True)

    # Convert 'Final_Marks' into a binary class (0 or 1)
    df['Class_Label'] = (df['Final_Marks'] <= 5).astype(int)

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
    
    return df, unique_errors


def prepare_train_test_sets(df):
    """Splits the dataset into training and testing subsets after scaling the features."""
    df.fillna(df.median(), inplace=True)  # Handle missing values
    
    X = df.drop(columns=['Class_Label'])  # Extract features
    y = df['Class_Label']  # Extract labels

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.3, random_state=42)


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


def plot_knn_accuracy(X_train, X_test, y_train, y_test):
    """Plots the classification accuracy for different k values (1 to 11)."""
    k_values = range(1, 12)
    accuracy_scores = []

    for k in k_values:
        knn_model = train_knn_classifier(X_train, y_train, k)
        accuracy, _ = test_knn_classifier(knn_model, X_test, y_test)
        accuracy_scores.append(accuracy)

    plt.figure(figsize=(10, 5))
    plt.plot(k_values, accuracy_scores, marker='o', linestyle='dashed', color='b')
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Classification Accuracy")
    plt.title("KNN Accuracy vs. Number of Neighbors")
    plt.xticks(k_values)
    plt.grid()
    plt.show()


def display_confusion_matrix(y_test, y_pred):
    """Generates and displays a confusion matrix for model evaluation."""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.title("Confusion Matrix")
    plt.show()
    
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


if __name__ == "__main__":
    file_path = '15-C.xlsx'

    # Load and preprocess data
    dataset = load_data(file_path)
    dataset = clean_data(dataset)
    dataset, error_types = process_error_types(dataset)

    print("Unique Errors Found:", error_types)

    X_train, X_test, y_train, y_test = prepare_train_test_sets(dataset)

    # Train a KNN classifier with k=3
    knn_model = train_knn_classifier(X_train, y_train, k=3)

    # Evaluate the model
    knn_accuracy, knn_predictions = test_knn_classifier(knn_model, X_test, y_test)
    print("KNN Accuracy (k=3):", knn_accuracy)
    print("KNN Predictions (k=3):", knn_predictions)

    # Make a single prediction for a test vector
    test_vector = X_test[0].reshape(1, -1)  # Reshape required for a single prediction
    predicted_class = knn_model.predict(test_vector)
    print(f"Predicted Class for a Sample Test Vector: {predicted_class[0]}")

    # Train a nearest-neighbor (NN) classifier (k=1) and compare
    nn_model = train_knn_classifier(X_train, y_train, k=1)
    nn_accuracy, _ = test_knn_classifier(nn_model, X_test, y_test)
    print("Nearest Neighbor (k=1) Accuracy:", nn_accuracy)

    # Generate accuracy plot for different k values
    plot_knn_accuracy(X_train, X_test, y_train, y_test)

    # Generate and display confusion matrix
    display_confusion_matrix(y_test, knn_predictions)

    # **Analysis:**
    if knn_accuracy < 0.6:
        print("The model is likely underfitting. Consider adding more features or using a more complex model.")
    elif knn_accuracy > 0.9 and nn_accuracy < 0.7:
        print("The model may be overfitting. Consider adding regularization or reducing k.")
    else:
        print("The model appears to be well-fitted.")
