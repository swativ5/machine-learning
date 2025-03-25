import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_data(filepath):
    """Load dataset from a CSV file."""
    df = pd.read_excel(filepath)
    return df

def split_data(df, target_column="Final_Marks", test_size=0.2, random_state=42):
    """Split dataset into features (X) and target (y), then into train and test sets."""
    X = df.drop(columns=[target_column])  # Features
    y = df[target_column]  # Target

    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def scale_features(X_train, X_test):
    """Normalize feature values using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_mlp_classifier(X_train, y_train, hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500):
    """Train an MLP classifier with the specified parameters."""
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, 
                        solver=solver, max_iter=max_iter, random_state=42)
    mlp.fit(X_train, y_train)
    return mlp

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model and print performance metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    filepath = "AST.xlsx"  # Change to your dataset path
    df = load_data(filepath)

    X_train, X_test, y_train, y_test = split_data(df)

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    model = train_mlp_classifier(X_train_scaled, y_train)

    evaluate_model(model, X_test_scaled, y_test)