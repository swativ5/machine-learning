import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

def load_data(file_path, sheet_name):
    # Load dataset from Excel file.
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    return data

def identify_columns(data):
    # Identify categorical and numerical columns.
    categorical_cols = [col for col in data.columns if data[col].dtype == 'object']
    numerical_cols = [col for col in data.columns if data[col].dtype != 'object']
    return categorical_cols, numerical_cols

def encode_categorical_data(data, categorical_cols):
    # Encode categorical columns using Label Encoding or One-Hot Encoding.
    encoded_data = data.copy()
    label_encoders = {}
    one_hot_columns = []

    for col in categorical_cols:
        unique_values = data[col].unique()
        if len(unique_values) <= 5:
            le = LabelEncoder()
            encoded_data[col] = le.fit_transform(data[col])
            label_encoders[col] = le
        else:
            one_hot_columns.append(col)

    encoded_data = pd.get_dummies(encoded_data, columns=one_hot_columns)
    return encoded_data, label_encoders

def check_missing_values(data):
    # Check for missing values in the dataset.
    missing_values = {}
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            missing_values[col] = data[col].isnull().sum()
    return missing_values

def calculate_ranges(data, numerical_cols):
    # Calculate the range (min, max) for numerical columns.
    ranges = {}
    for col in numerical_cols:
        ranges[col] = (data[col].min(), data[col].max())
    return ranges

def detect_outliers(data, numerical_cols):
    # Detect outliers using boxplots for numerical columns.
    outliers_dict = {}
    if numerical_cols:
        for col in numerical_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = data[col][(data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))]
            outliers_dict[col] = outliers.tolist()  # Store as a list for easy interpretation
        return outliers_dict
    else:
        print("No numerical columns to plot.")

def calculate_mean_variance(data, numerical_cols):
    # Calculate mean and variance for numerical columns.
    stats = {}
    for col in numerical_cols:
        stats[col] = {
            "mean": data[col].mean(),
            "variance": data[col].std() ** 2
        }
    return stats

def impute_missing_values(encoded_data):
    # Impute missing values using mean, median, or mode based on the data type and presence of outliers.
    numerical_cols, categorical_cols = identify_columns(encoded_data)
    for col in numerical_cols:
        if encoded_data[col].isnull().sum() > 0:
            if encoded_data[col].skew() > 1 or encoded_data[col].skew() < -1:  # Check for outliers using skewness
                encoded_data[col].fillna(encoded_data[col].median(), inplace=True)  # Use median for skewed data
                print(f"Imputed {col} with median: {encoded_data[col].median()}")
            else:
                data[col].fillna(encoded_data[col].mean(), inplace=True)  # Use mean for normal data
                print(f"Imputed {col} with mean: {encoded_data[col].mean()}")

    for col in categorical_cols:
        if encoded_data[col].isnull().sum() > 0:
            encoded_data[col].fillna(encoded_data[col].mode()[0], inplace=True)  # Use mode for categorical data
            print(f"Imputed {col} with mode: {encoded_data[col].mode()[0]}")

    return encoded_data

def normalize_data(data, numerical_cols):
    # Normalize numerical columns using Min-Max Scaling.
    scaler = MinMaxScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    print("Data normalized using Min-Max Scaling.")
    return data

def calculate_jaccard_smc(vector1, vector2):
    # Calculate Jaccard Coefficient (JC) and Simple Matching Coefficient (SMC) for binary vectors.
    # Convert vectors to binary (0 or 1)
    vector1 = np.where(pd.to_numeric(vector1, errors='coerce') > 0, 1, 0)
    vector2 = np.where(pd.to_numeric(vector2, errors='coerce') > 0, 1, 0)


    # Calculate f11, f10, f01, f00
    f11 = np.sum((vector1 == 1) & (vector2 == 1))
    f10 = np.sum((vector1 == 1) & (vector2 == 0))
    f01 = np.sum((vector1 == 0) & (vector2 == 1))
    f00 = np.sum((vector1 == 0) & (vector2 == 0))

    # Calculate Jaccard Coefficient (JC)
    jc = f11 / (f01 + f10 + f11) if (f01 + f10 + f11) != 0 else 0

    # Calculate Simple Matching Coefficient (SMC)
    smc = (f11 + f00) / (f00 + f01 + f10 + f11) if (f00 + f01 + f10 + f11) != 0 else 0

    return jc, smc

def calculate_cosine_similarity(vector1, vector2):
    # Calculate Cosine Similarity between two vectors.\
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    cosine_sim = dot_product / (norm_vector1 * norm_vector2) if (norm_vector1 * norm_vector2) != 0 else 0
    return cosine_sim

def plot_similarity_heatmaps(data, num_vectors=20):
    # Computes and plots heatmaps for Jaccard Coefficient, Simple Matching Coefficient (SMC), and Cosine Similarity for the first `num_vectors` observation vectors in `data`.    """
    # Ensure all values are numeric
    data_numeric = data.apply(pd.to_numeric, errors='coerce')
    subset_data = data_numeric.iloc[:num_vectors]

    # Initialize similarity matrices
    jc_matrix = np.zeros((num_vectors, num_vectors))
    smc_matrix = np.zeros((num_vectors, num_vectors))
    cosine_matrix = np.zeros((num_vectors, num_vectors))

    # Compute similarity values for each pair
    for i in range(num_vectors):
        for j in range(num_vectors):
            jc, smc = calculate_jaccard_smc(subset_data.iloc[i].values, subset_data.iloc[j].values)
            cos = calculate_cosine_similarity(subset_data.iloc[i].values, subset_data.iloc[j].values)

            jc_matrix[i, j] = jc
            smc_matrix[i, j] = smc
            cosine_matrix[i, j] = cos

    # Convert to DataFrames for visualization
    jc_df = pd.DataFrame(jc_matrix, index=range(1, num_vectors + 1), columns=range(1, num_vectors + 1))
    smc_df = pd.DataFrame(smc_matrix, index=range(1, num_vectors + 1), columns=range(1, num_vectors + 1))
    cos_df = pd.DataFrame(cosine_matrix, index=range(1, num_vectors + 1), columns=range(1, num_vectors + 1))

    # Plot Heatmaps
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    sns.heatmap(jc_df, annot=True, cmap="Blues", fmt=".2f")
    plt.title("Jaccard Coefficient")

    plt.subplot(1, 3, 2)
    sns.heatmap(smc_df, annot=True, cmap="Blues", fmt=".2f")
    plt.title("Simple Matching Coefficient")

    plt.subplot(1, 3, 3)
    sns.heatmap(cos_df, annot=True, cmap="Blues", fmt=".2f")
    plt.title("Cosine Similarity")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load the data
    file_path = "Lab Session Data.xlsx"
    sheet_name = "thyroid0387_UCI"
    data = load_data(file_path, sheet_name)
    print("Data loaded successfully.")
    print(data.head())

    # Identify categorical and numerical columns
    categorical_cols, numerical_cols = identify_columns(data)
    print("Categorical columns:", categorical_cols)
    print("Numerical columns:", numerical_cols)

    # Encode categorical data
    encoded_data, label_encoders = encode_categorical_data(data, categorical_cols)
    print("Data after encoding:")
    print(encoded_data.head())

    # Check for missing values
    missing_values = check_missing_values(encoded_data)
    if missing_values:
        print("Missing values found:")
        for col, count in missing_values.items():
            print(f"{col}: {count} missing values")
    else:
        print("No missing values found.")

    # Calculate ranges for numerical columns
    ranges = calculate_ranges(data, numerical_cols)
    print("Ranges for numerical columns:")
    for col, col_range in ranges.items():
        print(f"{col}: {col_range}")

    # Detect outliers using boxplots
    outliers = detect_outliers(encoded_data, numerical_cols)
    if outliers:
        print("Outliers detected:")
        for col, outlier_values in outliers.items():
            print(f"{col}: {outlier_values}")
    else:
        print("No outliers detected.")

    # Calculate mean and variance for numerical columns
    stats = calculate_mean_variance(encoded_data, numerical_cols)
    print("Mean and variance for numerical columns:")
    for col, values in stats.items():
        print(f"{col}: Mean = {values['mean']}, Variance = {values['variance']}")

    # Impute missing values
    encoded_data = impute_missing_values(encoded_data)
    print("Data after imputation:")
    print(encoded_data.head())

    # Normalize numerical columns
    data = normalize_data(encoded_data, numerical_cols)
    print("Data after normalization:")
    print(data.head())

    # Similarity Measure (Jaccard Coefficient and SMC)
    # Select the first two observation vectors with binary attributes
    binary_cols = [col for col in encoded_data.columns if encoded_data[col].nunique() == 2]  # Identify binary columns
    vector1 = encoded_data.iloc[0][binary_cols].values
    vector2 = encoded_data.iloc[1][binary_cols].values

    jc, smc = calculate_jaccard_smc(vector1, vector2)
    print(f"Jaccard Coefficient (JC): {jc}")
    print(f"Simple Matching Coefficient (SMC): {smc}")

    # Cosine Similarity Measure
    # Select the first two observation vectors with all attributes
    vector1_all = encoded_data.iloc[0].values
    vector2_all = encoded_data.iloc[1].values

    cosine_sim = calculate_cosine_similarity(vector1_all, vector2_all)
    print(f"Cosine Similarity: {cosine_sim}")

    # Plot similarity heatmaps
    plot_similarity_heatmaps(encoded_data)