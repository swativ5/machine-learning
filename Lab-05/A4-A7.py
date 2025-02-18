import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_data(file_path, target_column='Final_Marks'):
    """Loads dataset from a CSV file and removes the target variable."""
    logging.info("Loading dataset\n")
    df = pd.read_csv(file_path)
    
    if target_column in df.columns:
        df = df.drop(columns=[target_column])
        logging.info(f"Dropped target column: {target_column}\n")
    
    return df.values  # Return as NumPy array

def perform_kmeans(X, k):
    """Performs K-Means clustering with k clusters."""
    logging.info(f"Performing K-Means with k={k} clusters.\n")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
    return kmeans

def evaluate_clustering(X, kmeans):
    """Calculates clustering evaluation metrics."""
    logging.info("Evaluating clustering performance.\n")
    silhouette = silhouette_score(X, kmeans.labels_)
    ch_score = calinski_harabasz_score(X, kmeans.labels_)
    db_index = davies_bouldin_score(X, kmeans.labels_)

    logging.info(f"Silhouette Score: {silhouette:.4f}, CH Score: {ch_score:.4f}, DB Index: {db_index:.4f}\n")
    return {"Silhouette Score": silhouette, "CH Score": ch_score, "DB Index": db_index}

def optimal_k_using_scores(X, k_range=range(2, 10)):
    """Plots evaluation scores for different k values to determine the optimal k."""
    logging.info("Finding optimal k using evaluation scores.\n")
    sil_scores, ch_scores, db_scores = [], [], []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
        sil_scores.append(silhouette_score(X, kmeans.labels_))
        ch_scores.append(calinski_harabasz_score(X, kmeans.labels_))
        db_scores.append(davies_bouldin_score(X, kmeans.labels_))

    plt.figure(figsize=(10, 5))
    plt.plot(k_range, sil_scores, marker='o', label="Silhouette Score")
    plt.plot(k_range, ch_scores, marker='s', label="CH Score")
    plt.plot(k_range, db_scores, marker='^', label="DB Index")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Clustering Evaluation Metrics")
    plt.show()

def elbow_method(X, k_range=range(2, 20)):
    """Plots inertia for different k values to determine the optimal k using the Elbow Method."""
    logging.info("Finding optimal k using the Elbow Method.\n")
    distortions = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
        distortions.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, distortions, marker='o', linestyle='-')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal K")
    plt.show()

if __name__ == "__main__":
    # Load data
    file_path = "GST - C_AST.csv"
    X = load_data(file_path)

    # Initial clustering with k=2
    kmeans = perform_kmeans(X, k=2)
    evaluate_clustering(X, kmeans)

    # Find the optimal k
    optimal_k_using_scores(X)
    elbow_method(X)

    # Perform K-Means with chosen k (9 based on elbow method)
    optimal_k = 9
    kmeans_optimal = perform_kmeans(X, optimal_k)
    evaluate_clustering(X, kmeans_optimal)
