import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from lime.lime_tabular import LimeTabularExplainer

# -------------------------------
# Step 1: Load Dataset from Excel
# -------------------------------
df = pd.read_excel("AST.xlsx")

# Define the Target Column 
target_column = "Final_Marks" 

# Identify Features (X) and Target (y)
X = df.drop(columns=[target_column])  # All columns except the target
y = df[target_column]  # Target column

# Handle Missing Values
X.fillna(X.mean(), inplace=True)  # Fill missing values with column mean

# Encode Categorical Columns (if any)
for col in X.select_dtypes(include=['object']).columns:
    encoder = LabelEncoder()
    X[col] = encoder.fit_transform(X[col])

# Convert to NumPy Arrays for ML Models
X = X.values
y = y.values

# -------------------------------
# Step 2: Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# Step 3: Implement Stacking Model (Classifier or Regressor)
# -------------------------------
# Define if it's a Classification or Regression Problem
is_classification = len(set(y)) <= 10  # Assuming classification if target has ≤ 10 unique values

if is_classification:
    # Base Models for Classification
    base_models = [
        ('dt', DecisionTreeClassifier(max_depth=5, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
        ('svm', SVC(probability=True, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ]
    
    # Meta Model (Final Estimator)
    final_model = LogisticRegression()
    
    # Stacking Classifier
    stacking_model = StackingClassifier(estimators=base_models, final_estimator=final_model)

else:
    # Base Models for Regression
    base_models = [
        ('dt', DecisionTreeRegressor(max_depth=5)),
        ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=50)),
        ('svm', SVR()),
        ('knn', KNeighborsRegressor(n_neighbors=5))
    ]
    
    # Meta Model (Final Estimator)
    final_model = LinearRegression()
    
    # Stacking Regressor
    stacking_model = StackingRegressor(estimators=base_models, final_estimator=final_model)

# -------------------------------
# Step 4: Create Pipeline (Preprocessing + Stacking Model)
# -------------------------------
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardizing features
    ('model', stacking_model)      # Applying the Stacking Model
])

# Train the Pipeline
pipeline.fit(X_train, y_train)

# -------------------------------
# Step 5: Use LIME Explainer to Explain Model Predictions
# -------------------------------
explainer = LimeTabularExplainer(
    X_train,
    feature_names=[f'Feature {i}' for i in range(X.shape[1])],
    class_names=['Class 0', 'Class 1'] if is_classification else ["Target"],
    discretize_continuous=True
)

# Explain a Single Prediction
idx = 0  # Choose an index of a test sample
exp = explainer.explain_instance(X_test[idx], pipeline.predict_proba if is_classification else pipeline.predict)
exp.show_in_notebook()  # Show explanation

print("Model training and LIME explanation completed successfully!")