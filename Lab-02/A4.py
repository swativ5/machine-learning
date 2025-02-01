import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt

def load_data(file_path, sheet_name):
    # Load dataset from Excel file.
    return pd.read_excel(file_path, sheet_name=sheet_name)

def calculate_price_stats(data):
    # Calculate mean and variance of Price column
    price_data = data["Price"]
    mean_price = statistics.mean(price_data)
    variance_price = statistics.variance(price_data)
    return mean_price, variance_price

def sample_mean_wednesday(data, population_mean):
    # Select price data for all Wednesdays and calculate sample mean
    wednesday_data = data[data["Day"] == "Wed"]["Price"]
    mean_wednesday = statistics.mean(wednesday_data)
    return mean_wednesday, mean_wednesday - population_mean

def sample_mean_april(data, population_mean):
    # Select price data for April and calculate sample mean
    april_data = data[data["Month"] == "Apr"]["Price"]
    mean_april = statistics.mean(april_data)
    return mean_april, mean_april - population_mean

def probability_of_loss(data):
    # Calculate probability of making a loss
    chg_data = data["Chg%"]
    loss_days = sum(chg_data.apply(lambda x: x < 0))
    return loss_days / len(chg_data)

def probability_of_profit_wednesday(data):
    # Calculate probability of making a profit on Wednesday
    wednesday_data = data[data["Day"] == "Wed"]["Chg%"]
    profit_days = sum(wednesday_data.apply(lambda x: x > 0))
    return profit_days / len(wednesday_data)

def conditional_probability_profit_given_wednesday(data):
    # Calculate conditional probability of making a profit given that today is Wednesday
    prob_profit_wednesday = probability_of_profit_wednesday(data)
    prob_wednesday = len(data[data["Day"] == "Wed"]) / len(data)
    return prob_profit_wednesday / prob_wednesday if prob_wednesday > 0 else 0

def scatter_plot_chg_vs_day(data):
    # Create scatter plot of Chg% against the day of the week
    plt.figure(figsize=(10, 6))
    plt.scatter(data["Day"], data["Chg%"], alpha=0.5, color='purple')
    plt.xlabel("Day of the Week")
    plt.ylabel("Chg%")
    plt.title("Stock Price Change (%) vs. Day of the Week")
    plt.show()

if __name__ == "__main__":
    # Load data
    file_path = "Lab Session Data.xlsx"
    sheet_name = 'IRCTC Stock Price'
    data = load_data(file_path, sheet_name)

    # Compute statistics
    mean_price, variance_price = calculate_price_stats(data)
    print(f"Mean Price: {mean_price}, Variance: {variance_price}")

    # Sample means and comparisons
    mean_wednesday, diff_wednesday = sample_mean_wednesday(data, mean_price)
    print(f"Mean Price on Wednesdays: {mean_wednesday}, Difference from population mean: {diff_wednesday}")

    mean_april, diff_april = sample_mean_april(data, mean_price)
    print(f"Mean Price in April: {mean_april}, Difference from population mean: {diff_april}")

    # Probability calculations
    prob_loss = probability_of_loss(data)
    print(f"Probability of making a loss: {prob_loss}")

    prob_profit_wed = probability_of_profit_wednesday(data)
    print(f"Probability of making a profit on Wednesday: {prob_profit_wed}")

    cond_prob_profit_given_wed = conditional_probability_profit_given_wednesday(data)
    print(f"Conditional probability of profit given it's Wednesday: {cond_prob_profit_given_wed}")

    # Scatter plot
    scatter_plot_chg_vs_day(data)
