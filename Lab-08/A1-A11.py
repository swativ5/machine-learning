import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# A1: Define the perceptron components
def summation_unit(inputs, weights, bias):
    """Compute the weighted sum."""
    return np.dot(inputs, weights) + bias

def step_function(x):
    """Step activation function."""
    return 1 if x >= 0 else 0

def bipolar_step_function(x):
    """Bipolar step activation function."""
    return 1 if x >= 0 else -1

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def tanh_activation(x):
    """TanH activation function."""
    return np.tanh(x)

def relu(x):
    """ReLU activation function."""
    return max(0, x)

def leaky_relu_activation(x, alpha=0.01):
    """Leaky ReLU activation function."""
    return x if x > 0 else alpha * x

def comparator_unit(actual, predicted):
    """Comparator unit to compute error."""
    return actual - predicted

# Perceptron Training Function (used in A2, A3, A4, A5, A6)
def train_perceptron(X, y, activation_func, epochs=1000, lr=0.05, initial_weights=None, initial_bias=None):
    if initial_weights is None:
        # Default random initialization if not provided
        np.random.seed(0)
        weights = np.random.randn(X.shape[1])
    else:
        weights = np.array(initial_weights, dtype=float)
    
    if initial_bias is None:
        # Random bias if not provided
        np.random.seed(0)
        bias = np.random.randn()
    else:
        bias = float(initial_bias)
    
    errors = []
    
    for epoch in range(epochs):
        total_error = 0
        for i in range(len(X)):
            # Compute weighted sum
            linear_output = summation_unit(X[i], weights, bias)
            # Apply activation function
            y_pred = activation_func(linear_output)
            error = y[i] - y_pred
            total_error += error**2
            
            # Update weights and bias (gradient descent update rule)
            weights += lr * error * X[i]
            bias += lr * error
            
        errors.append(total_error)
        if total_error <= 0.002:
            break  # Convergence criterion met
    return weights, bias, errors

# A2: Implement AND gate perceptron training with given initial weights and bias
def train_and_gate():
    data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels = np.array([0, 0, 0, 1])
    # Use the provided initial weights and bias: W0 = 10, W1 = 0.2, W2 = -0.75
    initial_weights = [0.2, -0.75]
    initial_bias = 10
    return train_perceptron(data, labels, step_function, lr=0.05, initial_weights=initial_weights, initial_bias=initial_bias)

# A3: Train perceptron with different activation functions (for AND gate)
def train_with_different_activations():
    data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels = np.array([0, 0, 0, 1])
    
    # For these experiments we use random initialization unless specified otherwise.
    step_results = train_perceptron(data, labels, step_function)
    bipolar_step_results = train_perceptron(data, labels, bipolar_step_function)
    sigmoid_results = train_perceptron(data, labels, sigmoid)
    relu_results = train_perceptron(data, labels, relu)
    
    return step_results, bipolar_step_results, sigmoid_results, relu_results

# A4: Varying learning rate for AND gate perceptron training
def train_with_varying_lr():
    learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    iterations = []
    
    data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels = np.array([0, 0, 0, 1])
    
    for lr in learning_rates:
        # Using the fixed initial weights for consistency in AND gate training
        _, _, errors = train_perceptron(data, labels, step_function, lr=lr, initial_weights=[0.2, -0.75], initial_bias=10)
        iterations.append(len(errors))
    
    plt.plot(learning_rates, iterations, marker='o')
    plt.xlabel("Learning Rate")
    plt.ylabel("Iterations to Converge")
    plt.title("Effect of Learning Rate on Convergence (AND Gate)")
    plt.show()

# A5: Train XOR Gate perceptron (note: perceptron may not converge for XOR)
def train_xor_gate():
    data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels = np.array([0, 1, 1, 0])
    return train_perceptron(data, labels, step_function)

# A6: Classify customer transactions using perceptron with sigmoid activation
def classify_transactions():
    data = np.array([
        [20, 6, 2, 386], [16, 3, 6, 289], [27, 6, 2, 393],
        [19, 1, 2, 110], [24, 4, 2, 280], [22, 1, 5, 167],
        [15, 4, 2, 271], [18, 4, 2, 274], [21, 1, 4, 148],
        [16, 2, 4, 198]
    ])
    labels = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    return train_perceptron(data, labels, sigmoid)

# A7: Compare perceptron with matrix pseudo-inverse on customer transactions
def compare_pseudo_inverse():
    data = np.array([
        [20, 6, 2, 386], [16, 3, 6, 289], [27, 6, 2, 393],
        [19, 1, 2, 110], [24, 4, 2, 280], [22, 1, 5, 167],
        [15, 4, 2, 271], [18, 4, 2, 274], [21, 1, 4, 148],
        [16, 2, 4, 198]
    ])
    labels = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    return np.dot(np.linalg.pinv(data), labels)

# A8-A9: Backpropagation using MLPClassifier for AND and XOR gates
def train_backpropagation():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([0, 0, 0, 1])
    y_xor = np.array([0, 1, 1, 0])
    
    and_model = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', 
                              learning_rate_init=0.05, max_iter=1000, random_state=0)
    xor_model = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', 
                              learning_rate_init=0.05, max_iter=1000, random_state=0)
    
    and_model.fit(X, y_and)
    xor_model.fit(X, y_xor)
    
    return and_model, xor_model

# A10-A11: Train MLPClassifier for AND and XOR gates (similar to A8-A9)
def train_mlp_and_xor():
    return train_backpropagation()

# A12: Train MLPClassifier on a custom customer transactions dataset
def train_mlp_custom_dataset():
    X = np.array([
        [20, 6, 2, 386], [16, 3, 6, 289], [27, 6, 2, 393],
        [19, 1, 2, 110], [24, 4, 2, 280], [22, 1, 5, 167],
        [15, 4, 2, 271], [18, 4, 2, 274], [21, 1, 4, 148],
        [16, 2, 4, 198]
    ])
    y = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    model = MLPClassifier(hidden_layer_sizes=(4,), activation='logistic', 
                          learning_rate_init=0.05, max_iter=1000, random_state=0)
    model.fit(X, y)
    return model

if __name__ == "__main__":
    # A2: AND gate perceptron training with specified initial weights
    and_weights, and_bias, and_errors = train_and_gate()
    print("AND Gate Training completed in {} epochs.".format(len(and_errors)))
    
    # A5: XOR gate perceptron training
    xor_weights, xor_bias, xor_errors = train_xor_gate()
    print("XOR Gate Training completed in {} epochs.".format(len(xor_errors)))
    
    # A7: Comparison using matrix pseudo-inverse
    pseudo_inverse_result = compare_pseudo_inverse()
    print("Pseudo-inverse weights (customer transactions):", pseudo_inverse_result)
    
    # A8-A9: Backpropagation for AND and XOR gates using MLPClassifier
    mlp_and_model, mlp_xor_model = train_backpropagation()
    print("MLP AND gate predictions:", mlp_and_model.predict([[0, 0], [0, 1], [1, 0], [1, 1]]))
    print("MLP XOR gate predictions:", mlp_xor_model.predict([[0, 0], [0, 1], [1, 0], [1, 1]]))
    
    # Plot error convergence for perceptron training on logic gates (from A2 and A5)
    plt.plot(and_errors, label="AND Gate")
    plt.plot(xor_errors, label="XOR Gate (likely non-convergent)")
    plt.xlabel("Epochs")
    plt.ylabel("Sum-Square Error")
    plt.legend()
    plt.title("Error Convergence for AND & XOR Gates (Perceptron)")
    plt.show()
