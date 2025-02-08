import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski
from sklearn.preprocessing import StandardScaler

def clean_data(df):
    """
    Removes unnecessary columns and converts 'Final_Marks' into a binary classification label.
    """
    columns_to_remove = ['Question', 'Correct_Code', 'Code_with_Error', 'code_processed',
                         'code_with_question', 'code_comment', 'code_with_solution', 'ast']
    df.drop(columns=columns_to_remove, inplace=True)

    # Convert 'Final_Marks' into a binary target: 1 (≤5), 0 (>5)
    df['Grade'] = (df['Final_Marks'] <= 5).astype(int)

    return df

def encode_error_types(df):
    """
    Extracts unique error types from 'Type_of_Error' column and applies one-hot encoding.
    """
    unique_errors = set()
    for entry in df['Type_of_Error']:
        error_list = entry.strip("[]").replace("'", "").split(", ")  # Clean formatting
        unique_errors.update(error_list)

    # Create a binary column for each error type
    for error in unique_errors:
        df[error] = df['Type_of_Error'].apply(lambda x: 1 if error in x else 0)

    df.drop(columns=['Type_of_Error'], inplace=True)
    
    return df

def load_and_process_data(file_path):
    """
    Loads the dataset, cleans it, and encodes error types.
    """
    df = pd.read_excel(file_path)
    df = clean_data(df)
    df = encode_error_types(df)
    return df

def extract_features_and_target(df):
    """
    Extracts feature variables and target labels, handling missing values and scaling features.
    """
    df.fillna(df.median(), inplace=True)  # Replace NaNs with median values
    
    features = df.drop(columns=['Grade'])  # Independent variables
    labels = df['Grade']  # Target variable

    # Normalize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    return scaled_features, labels

def compute_centroids_and_variance(features, labels, class_0, class_1):
    """
    Computes the centroids and spread (standard deviation) of feature vectors for two given classes.
    """
    vectors_class_0 = features[labels == class_0]
    vectors_class_1 = features[labels == class_1]
    
    centroid_0 = np.mean(vectors_class_0, axis=0)
    centroid_1 = np.mean(vectors_class_1, axis=0)
    
    spread_0 = np.std(vectors_class_0, axis=0)
    spread_1 = np.std(vectors_class_1, axis=0)
    
    # Compute interclass distance
    interclass_distance = np.linalg.norm(centroid_0 - centroid_1)
    
    return centroid_0, spread_0, centroid_1, spread_1, interclass_distance

def analyze_feature_distribution(df, feature_index):
    """
    Analyzes a given feature by computing its mean, variance, and displaying a histogram.
    """
    feature_values = df.iloc[:, feature_index].dropna().values  # Remove NaN values
    
    mean_value = np.mean(feature_values)
    variance_value = np.var(feature_values)
    
    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(feature_values, bins=10, edgecolor='black', alpha=0.7)
    plt.xlabel("Feature Values")
    plt.ylabel("Frequency")
    plt.title(f"Histogram for Feature: {df.columns[feature_index]}")
    plt.show()
    
    return mean_value, variance_value

def minkowski_distance_analysis(features, index_1, index_2, r_values):
    """
    Computes Minkowski distance between two feature vectors for various r values.
    """
    vec_1 = features[index_1]
    vec_2 = features[index_2]
    
    distances = [minkowski(vec_1, vec_2, r) for r in r_values]
    
    # Plot Minkowski distance vs r values
    plt.figure(figsize=(8, 5))
    plt.plot(r_values, distances, marker='o', linestyle='-', color='b')
    plt.xlabel("r Value")
    plt.ylabel("Minkowski Distance")
    plt.title("Minkowski Distance vs r")
    plt.xticks(r_values)
    plt.grid()
    plt.show()
    
    return distances

if __name__ == "__main__":
    # Load and process dataset
    file_path = "15-C.xlsx"  # Update with actual file path
    df = load_and_process_data(file_path)
    
    # Extract features and labels
    features, labels = extract_features_and_target(df)
    
    # Define class labels for comparison
    class_0, class_1 = 0, 1  # Class 0: High marks (>5), Class 1: Low marks (≤5)
    
    # Compute centroids, spread, and interclass distance
    centroid_0, spread_0, centroid_1, spread_1, interclass_distance = compute_centroids_and_variance(
        features, labels, class_0, class_1
    )
    
    # Display results
    print(f"\nCentroid for Class {class_0}:\n", centroid_0)
    print(f"\nSpread (Standard Deviation) for Class {class_0}:\n", spread_0)
    print(f"\nCentroid for Class {class_1}:\n", centroid_1)
    print(f"\nSpread (Standard Deviation) for Class {class_1}:\n", spread_1)
    print(f"\nInterclass Distance between Class {class_0} and Class {class_1}: {interclass_distance:.4f}")
    
    # Analyze a specific feature
    selected_feature_index = 5  # Modify as needed
    mean_val, variance_val = analyze_feature_distribution(df, selected_feature_index)
    
    print(f"\nFeature Analyzed: {df.columns[selected_feature_index]}")
    print(f"\nMean: {mean_val:.4f}")
    print(f"\nVariance: {variance_val:.4f}")
    
    # Compute Minkowski distances
    vector_1_index, vector_2_index = 0, 1  # Modify indices if needed
    r_values_range = np.arange(1, 11)
    minkowski_distances = minkowski_distance_analysis(features, vector_1_index, vector_2_index, r_values_range)
    
    # Display Minkowski distances
    for r, dist in zip(r_values_range, minkowski_distances):
        print(f"r = {r}: Minkowski Distance = {dist:.4f}")
