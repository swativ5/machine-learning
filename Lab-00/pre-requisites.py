import numpy as np

# Q1. Create a vector, V1, (length = 100) of random number. Sort them in increasing order.
v1 = np.random.random(100)
print(v1)
v1_sorted = np.sort(v1)
print(v1_sorted)

# Q2. Find out what happens when perform V1*3. Multiply V1 with 3 to scale each value 3 times.
v2 = v1 * 3
print(v2)

# Q3. Find the mean and standard deviation for the vector.
print("mean:", np.average(v2), np.mean(v2))
print("variance", np.var(v2))
print("standard deviation:", np.std(v2))

# 4. Make a matrix of zeros with 4 rows and 3 columns. Fill the matrix with random numbers.
#  Convert the matrix into a single dimensional array.
matrix4 = np.zeros((4, 3))
print(matrix4)
random_matrix = np.array([[np.random.random() for i in range(4)] for j in range(3)])
print(random_matrix)
single_matrix = random_matrix.flatten()
print(single_matrix)

# Q5. Take a string {S1 = “I am a great learner. I am going to have an awesome life.”}. 
# Search for the substring “Am” and find the occurrence count.
s1 = "I am a great learner. I am going to have an awesome life."
print(s1.count("am"))

# Q6. S2 = “I work hard and shall be rewarded well”. Add both the strings to make a single string S3.
s2 = "I work hard and shall be rewarded well"
s3 = s1 + s2
print(s3)

# Q7. Taking the above string S3, split the string into words by using “white space” and “period(.)” 
# as splitters. Put the words into an array. Find the length of the array.
import re
array = re.split(r'[.\s]+', s3)
array = [a for a in array if a]
print(array)
print(len(array))

# Q8. Remove words “I”, “Am”, “to” & “and” from the array. 
# Also, remove words containing more than 6 characters. Find the length of the array.
words = set(("I", "Am", "to", "and"))
array = [word for word in array if word not in words]
print(array)
array = [word for word in array if len(word) <= 6]
print(array)

# Q9. Consider the date “01-JUN-2021”. 
# Split the date and find the values for date, month and year separately. 
# Find the numerical value for the month using calendar sequence (Jan = 1, Dec = 12).
date_input = "01-JUN-2021"
date, month, year = date_input.split("-")
month_map = {
    "JAN": 1, 
    "FEB": 2, 
    "MAR": 3, 
    "APR": 4, 
    "MAY": 5, 
    "JUN": 6,
    "JUL": 7, 
    "AUG": 8, 
    "SEP": 9, 
    "OCT": 10, 
    "NOV": 11, 
    "DEC": 12
}
print("date:", date)
print("month:", month, month_map[month])
print("year:", year)

# Q10. Create an excel file with 3 columns. The 3 columns contain the names of city, state and PIN code.
# Make 10 row entries for 10 cities. (Refer table below.) Load this excel file into a data table. Make a
# fourth column that contains a string with the city & state joined with a comma separation. Write the
# table back into an excel file.
import pandas as pd
data = data = {
    'City': ['BENGALURU', 'CHENNAI', 'MUMBAI', 'MYSURU', 'PATNA', 'JAMMU', 'GANDHI NAGAR', 'HYDERABAD', 'ERNAKULAM', 'AMARAVATI'],
    'State': ['KA', 'TN', 'MH', 'KA', 'BH', 'JK', 'GJ', 'TS', 'KL', 'AP'],
    'PIN Code': [560001, 600001, 400001, 570001, 800001, 180001, 382001, 500001, 682001, 522001]
}
df = pd.DataFrame(data)
df["City, State"] = df["City"] + ", " + df["State"]
df.to_excel("states.xlsx", index = False)
print(df)

# Q11. Sort the vector V1 in increasing order and plot it. 
# Observe the color of the plot & change the color of the plot to red.
import matplotlib.pyplot as plt
v1 = np.random.random(100)
v1 = np.sort(v1)
# plt.plot(v1, color="red")
# plt.show()

# Q12. Create another vector V2 containing the square of the number present in V1. 
# Make a plot of both V1 and V2.
v2 = v1 ** 2
plt.plot(v1, color='blue')
plt.plot(v2, color='green')
plt.show()