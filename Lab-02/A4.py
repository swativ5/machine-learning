import numpy as np
import pandas as pd
import statistics

file_path = "Lab Session Data.xlsx"
sheet_name = 'IRCTC Stock Price'
data = pd.read_excel(file_path, sheet_name=sheet_name)


# Calculate the mean and variance of the Price data present in column D. 
D = np.matrix(data[["Price"]])
pmean = np.mean(D)
print(np.mean(D))
print(np.std(D))

w_days = data[data["Day"] == "Wed"]
w_price = w_days["Price"]
w_mean = np.mean(w_price)
print(w_mean, pmean)
# he slightly lower mean on Wednesdays suggests that the average price of purchases on Wednesdays tends to be a bit less than the overall average for the entire dataset. This could be due to specific sales or promotions that happen on Wednesdays, or it could just be random variation.

a_month = data[data["Month"] == "Apr"]
a_price = a_month["Price"]
a_mean = np.mean(a_price)
print(a_mean, pmean)
# This indicates that, on average, the prices for purchases in April are higher compared to the prices for purchases across all other months. It could be due to seasonal factors, promotions, or specific market conditions that led to higher-priced transactions in April.


chg_v = data[["Chg%"]]
loss_days = chg_v[chg_v.apply(lambda x: x < 0)].count()
probabiity = loss_days/(chg_v.count())

w_days = data[data["Day"] == "Wed"]
w_chg = w_days["Chg%"]
w_profit = w_chg[w_chg.apply(lambda x: x < 0)].count()

