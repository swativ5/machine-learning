import numpy as np

def equal_width_binning(target_values, num_bins=4):
    """ Bins the continuous final marks into equal-width bins. """
    min_val, max_val = min(target_values), max(target_values)
    bin_width = (max_val - min_val) / num_bins
    bins = [min_val + i * bin_width for i in range(num_bins + 1)]
    bin_indices = np.digitize(target_values, bins, right=True)
    return bin_indices, bins

def calculate_entropy(bin_indices):
    """ Computes entropy given bin indices. """
    unique, counts = np.unique(bin_indices, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

