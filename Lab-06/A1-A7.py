import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# A1.
# Calculate the entropy of your dataset. Employ equal width binning and divide data into 4 bins. 
# Each bin may be considered as a categorical data value.
def bin_data(values, bins=4, method='equal_width'):
    # Simple equal-width binning
    if method == 'equal_width':
        return pd.cut(values, bins=bins, labels=False, include_lowest=True)
    # Frequency binning, if needed
    elif method == 'equal_freq':
        return pd.qcut(values, q=bins, labels=False, duplicates='drop')
    return values

def calculate_entropy(data, bins=4, binning_method='equal_width'):
    # Convert continuous data to categorical bins
    binned_data = bin_data(data, bins=bins, method=binning_method)
    # Compute frequency
    counts = np.bincount(binned_data)
    probs = counts[counts > 0] / len(binned_data)
    # Calculate entropy
    return -np.sum(probs * np.log2(probs))

# A2.
# Calculate the entropy of the Gini index value.
def calculate_gini_index(data):
    # Binning not strictly required here, but can be done if needed
    _, counts = np.unique(data, return_counts=True)
    probs = counts / len(data)
    gini = 1 - np.sum(probs**2)
    return gini

def entropy_of_gini(data):
    gini_val = calculate_gini_index(data)
    # Treat Gini value as if it were a probability distribution
    # (Though not strictly standard, we provide a basic approach)
    if gini_val <= 0 or gini_val >= 1:
        return 0.0
    return - (gini_val * np.log2(gini_val) + (1 - gini_val) * np.log2(1 - gini_val))

# A3.
# Create a function to detect the feature for the root note of a decision tree. Use information gain 
# as the impurity measure for identifying the root node. Assume the features to be categorical 
# or could be converted to categorical data by binning.

def info_gain(df, feature, target_col, bins=4, bin_method='equal_width'):
    # Entropy of the entire dataset
    total_entropy = calculate_entropy(df[target_col].values, bins, bin_method)
    # Compute weighted entropy after splitting on feature
    if df[feature].dtype.kind in ['f','i']:
        df['temp_binned'] = bin_data(df[feature], bins=bins, method=bin_method)
        feature_vals = df['temp_binned']
    else:
        feature_vals = df[feature]
    unique_vals = feature_vals.unique()
    weighted_entropy = 0.0
    for val in unique_vals:
        subset = df.loc[feature_vals == val]
        subset_entropy = calculate_entropy(subset[target_col].values, bins, bin_method)
        weighted_entropy += (len(subset) / len(df)) * subset_entropy
    return total_entropy - weighted_entropy

def detect_root_feature(df, features, target_col, impurity_measure='entropy', bins=4, bin_method='equal_width'):
    # Returns the feature with the highest information gain
    best_feature = None
    best_gain = -1
    for f in features:
        current_gain = info_gain(df.copy(), f, target_col, bins, bin_method)
        if current_gain > best_gain:
            best_gain = current_gain
            best_feature = f
    return best_feature

# A4. If the feature is continuous valued for A3, use equal width or frequency binning 
# for converting the attribute to categorical valued. The binning type should be a parameter
# to the function built for binning. Write your own function for the binning task. 
# The number of bins to be created should also be passed as a parameter to the function.
# Use function overloading to allow for usage of default parameters if no parameters are passed. 
# Demonstration of function overloading style via default parameters in Python
def bin_data(values, bins=4, method='equal_width'):
    """
    Bins continuous data into categorical values.
    bins : int (default 4)
    method : 'equal_width' or 'equal_freq' (default 'equal_width')
    """
    if method == 'equal_width':
        return pd.cut(values, bins=bins, labels=False, include_lowest=True)
    elif method == 'equal_freq':
        return pd.qcut(values, q=bins, labels=False, duplicates='drop')
    return values

# Example usage of default parameters (bins=4, method='equal_width')
def bin_data_default(values):
    return bin_data(values)

# Example usage with custom bins only
def bin_data_custom_bins(values, bins):
    return bin_data(values, bins=bins)

# Example usage with both custom bins and method
def bin_data_custom(values, bins, method):
    return bin_data(values, bins=bins, method=method)

# A5. 
# Expand the above functions to build your own Decision Tree module. 
class DecisionTree:
    def __init__(self, target_col, bins=4, bin_method='equal_width'):
        self.target_col = target_col
        self.bins = bins
        self.bin_method = bin_method
        self.tree = None

    def fit(self, df, features):
        # Simple recursion stopping conditions omitted for brevity
        root = detect_root_feature(df, features, self.target_col, bins=self.bins, bin_method=self.bin_method)
        self.tree = {root: {}}
        # Perform a simple split for demonstration
        if df[root].dtype.kind in ['f','i']:
            df['temp_binned'] = bin_data(df[root], bins=self.bins, method=self.bin_method)
            feature_vals = df['temp_binned']
        else:
            feature_vals = df[root]
        for val in feature_vals.unique():
            subset = df.loc[feature_vals == val].copy()
            self.tree[root][val] = subset[self.target_col].mode()[0]

    def predict(self, x):
        # Simple lookup if root only
        root = next(iter(self.tree))
        val = x[root]
        if isinstance(val, (float, int)):
            val = bin_data(pd.Series([val]), bins=self.bins, method=self.bin_method)[0]
        return self.tree[root].get(val, None)
    
# A6. 
# Draw and visualize the decision tree constructed based on your data. 
def visualize_decision_tree(tree):
    """
    Draw and visualize the decision tree constructed based on your data.
    'tree' is expected to be a dictionary object from the DecisionTree class above.
    This implementation uses matplotlib to render a simple tree diagram.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')

    def add_nodes(subtree, parent_name=None, depth=0, pos=(0.5, 1.0), width=1.0):
        if not isinstance(subtree, dict):
            ax.text(pos[0], pos[1], f"Class: {subtree}", ha='center', va='center', fontsize=12, bbox=dict(facecolor='lightblue', edgecolor='black'))
        else:
            root_feature = next(iter(subtree))
            ax.text(pos[0], pos[1], f"Feature: {root_feature}", ha='center', va='center', fontsize=12, bbox=dict(facecolor='lightgreen', edgecolor='black'))
            num_branches = len(subtree[root_feature])
            for i, (val, branch) in enumerate(subtree[root_feature].items()):
                new_pos = (pos[0] - width/2 + i * (width / num_branches), pos[1] - 0.2)
                ax.plot([pos[0], new_pos[0]], [pos[1], new_pos[1]], 'k-')
                add_nodes(branch, root_feature, depth + 1, new_pos, width / num_branches)

    root_feature = next(iter(tree))
    add_nodes(tree)

    plt.show()

# A7. 
# Use 2 features from your dataset for a classification problem. Visualize the decision boundary created by your DT in the vector space. 
def visualize_decision_boundary(model, df, feature1, feature2, target_col, resolution=0.01):
    """
    Use 2 features from your dataset for a classification problem. 
    Visualize the decision boundary created by your Decision Tree in the vector space.
    """
    x_min, x_max = df[feature1].min() - 1, df[feature1].max() + 1
    y_min, y_max = df[feature2].min() - 1, df[feature2].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution)
    )

    # Predict over the mesh
    Z = []
    for (a, b) in zip(xx.ravel(), yy.ravel()):
        sample_point = {feature1: a, feature2: b}
        Z.append(model.predict(sample_point))
    Z = np.array(Z).reshape(xx.shape)

    # Plot contour
    plt.figure(figsize=(6, 6))
    plt.contourf(xx, yy, Z, alpha=0.6, cmap=plt.cm.coolwarm)

    # Plot actual data
    plt.scatter(df[feature1], df[feature2], c=df[target_col], edgecolors='k', cmap=plt.cm.coolwarm)
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title("Decision Boundary for Two Features")
    plt.show()


if __name__ == "__main__":
    # Load the CSV dataset (replace the path as needed)
    df = pd.read_csv('GST - C_AST.csv', nrows = 100)
    
    # Select two features for classification 
    feature1 = 'ast_embedding_0'
    feature2 = 'ast_embedding_1'
    target_col = 'Final_Marks'  
    
    # Create a Decision Tree model
    tree_model = DecisionTree(target_col=target_col, bins=4, bin_method='equal_width')
    
    # Fit the model using your selected features
    tree_model.fit(df, [feature1, feature2])
    
    # Visualize the generated tree
    visualize_decision_tree(tree_model.tree)
    
    # Visualize the decision boundary
    visualize_decision_boundary(tree_model, df, feature1, feature2, target_col)
