import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Explainability modules
from lime.lime_tabular import LimeTabularExplainer
import shap

# Data Loading and Preprocessing Functions
def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from an Excel file.
    """
    return pd.read_excel(file_path)

def extract_features_and_target(df: pd.DataFrame) :
    """
    For the regression task, select the five mark columns as features and
    'Total Marks' as the target.
    
    Excluded non-predictive columns such as S.no., Question, code fields, etc.
    """
    feature_cols = [
        'Header_and_Main_Declaration(Marks)', 
        'Incomprehensible_Code(Marks)', 
        'Comprehensible_Code_with_logical_errors(Marks)', 
        'Comprehensible_code_with_syntax_errors(Marks)', 
        'Correct_code_and_output(Marks)'
    ]
    target_col = "Total Marks"
    X = df[feature_cols]
    y = df[target_col]
    return X, y

# A1. Feature Correlation Analysis (Heatmap)
def plot_feature_correlation(df: pd.DataFrame) -> None:
    """
    Computes and plots a heatmap of the correlation matrix for the given dataframe.
    Here, it is assumed that df contains only the numeric mark features.
    """
    corr_matrix = df.corr()
    print("Correlation Matrix:")
    print(corr_matrix)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
    plt.title("Correlation Heatmap of Mark Features")
    plt.tight_layout()
    plt.show()

# A2 & A3. PCA Experiments
def run_pca_experiment(X_train: np.ndarray, X_test: np.ndarray, y_train: pd.Series, y_test: pd.Series,
                       variance_ratio: float):
    """
    Performs PCA on the scaled features to retain the specified variance_ratio,
    trains a LinearRegression model on the transformed features, and returns the model and R² score.
    
    Parameters:
        variance_ratio (float): e.g., 0.99 or 0.95 for 99% or 95% of the explained variance.
    """
    pca = PCA(n_components=variance_ratio)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca  = pca.transform(X_test)
    
    print(f"\nPCA retaining {int(variance_ratio*100)}% variance:")
    print("  Number of components selected:", pca.n_components_)
    
    model = LinearRegression()
    model.fit(X_train_pca, y_train)
    
    preds = model.predict(X_test_pca)
    score = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    
    print(f"  R² Score: {score:.2f}")
    print(f"  MSE: {mse:.2f}")
    
    return model, score

# A4. Sequential Feature Selection using RFE
def run_rfe_experiment(X_train: np.ndarray, X_test: np.ndarray, y_train: pd.Series, y_test: pd.Series,
                       n_features_to_select: int):
    """
    Applies Recursive Feature Elimination (RFE) using a LinearRegression estimator,
    trains on the selected features, and returns the model and R² score.
    
    Parameters:
        n_features_to_select (int): Number of features to select.
    """
    base_estimator = LinearRegression()
    selector = RFE(estimator=base_estimator, n_features_to_select=n_features_to_select, step=1)
    selector.fit(X_train, y_train)
    
    print("\nRFE Feature Selection:")
    print("  Selected features (mask):", selector.support_)
    print("  Feature ranking:", selector.ranking_)
    
    X_train_rfe = selector.transform(X_train)
    X_test_rfe  = selector.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_rfe, y_train)
    preds = model.predict(X_test_rfe)
    score = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    
    print(f"  RFE Model R² Score: {score:.2f}")
    print(f"  RFE Model MSE: {mse:.2f}")
    
    return model, score

# A5. Explainability with LIME and SHAP
import matplotlib.pyplot as plt

def explain_with_lime_shap(X_train: np.ndarray, X_test: np.ndarray, baseline_model, feature_names: list):
    """
    Use LIME and SHAP to explain the baseline regression model predictions.
    Works in VS Code and other script environments.
    """
    # ------ LIME Explanation ------
    explainer_lime = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        mode='regression',
        discretize_continuous=True
    )

    instance_index = 0
    lime_exp = explainer_lime.explain_instance(
        X_test[instance_index], baseline_model.predict, num_features=10
    )

    print("\nLIME Explanation for test instance index", instance_index)

    fig = lime_exp.as_pyplot_figure()
    plt.title(f"LIME Explanation - Test Instance {instance_index}")
    plt.tight_layout()
    plt.show()  # ✅ This will render the plot in VS Code

    # ------ SHAP Explanation ------
    explainer_shap = shap.Explainer(baseline_model.predict, X_train)
    shap_values = explainer_shap(X_test)

    print("\nSHAP Summary Plot:")
    shap.summary_plot(shap_values, X_test, feature_names=feature_names)
    plt.title("SHAP Summary Plot")

# Main Function: Run All Experiments
if __name__ == "__main__":
    # Data Loading and Preprocessing
    file_path = "PythonMultiFunction.xlsx"
    df = load_dataset(file_path)
    
    # For A1, we perform correlation analysis on the mark columns.
    mark_cols = [
        'Header_and_Main_Declaration(Marks)', 
        'Incomprehensible_Code(Marks)', 
        'Comprehensible_Code_with_logical_errors(Marks)', 
        'Comprehensible_code_with_syntax_errors(Marks)', 
        'Correct_code_and_output(Marks)', 
        'Total Marks'
    ]
    df_marks = df[mark_cols]
    plot_feature_correlation(df_marks)
    
    # Prepare Modeling Features
    # For the regression task, we use the five mark sub-scores as features.
    X, y = extract_features_and_target(df)
    # Remove the target from features if accidentally included (here, only five features are used).
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    
    # Baseline Model on Original Scaled Features
    print("\n--- Baseline Model (Original Features) ---")
    baseline_model = LinearRegression()
    baseline_model.fit(X_train_scaled, y_train)
    preds = baseline_model.predict(X_test_scaled)
    baseline_r2 = r2_score(y_test, preds)
    baseline_mse = mean_squared_error(y_test, preds)
    print(f"Baseline R² Score: {baseline_r2:.2f}")
    print(f"Baseline MSE: {baseline_mse:.2f}")
    
    # A2. PCA Experiment: Retaining 99% Variance
    model_pca99, r2_pca99 = run_pca_experiment(X_train_scaled, X_test_scaled, y_train, y_test, variance_ratio=0.99)
    
    # A3. PCA Experiment: Retaining 95% Variance
    model_pca95, r2_pca95 = run_pca_experiment(X_train_scaled, X_test_scaled, y_train, y_test, variance_ratio=0.95)
    
    # A4. Sequential Feature Selection via RFE
    # For demonstration, select half of the features (ensure at least one feature is selected)
    n_features = X_train_scaled.shape[1]
    n_select = max(1, n_features // 2)
    model_rfe, r2_rfe = run_rfe_experiment(X_train_scaled, X_test_scaled, y_train, y_test, n_features_to_select=n_select)
    
    # A5. Explainability using LIME and SHAP
    feature_names = X.columns.tolist()
    explain_with_lime_shap(X_train_scaled, X_test_scaled, baseline_model, feature_names)
