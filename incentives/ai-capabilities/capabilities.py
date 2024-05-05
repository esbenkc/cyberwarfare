import pandas as pd

import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv("./Epoch Database - Notable Systems.csv")
# This is the output for an example row, filter the data for US and China
# 'United Kingdom of Great Britain and Northern Ireland,Canada'
# So split it up and filter for US and China
data["Country (from Organization)"] = data["Country (from Organization)"].str.split(",")
data["Country (from Organization)"] = data["Country (from Organization)"].apply(
    lambda x: (
        [country.strip() for country in x if country in ["US", "China"]][0]
        if x
        else None
    )
)

# Filter the data for US and China
us_data = data[data["Country (from Organization)"] == "US"]
china_data = data[data["Country (from Organization)"] == "China"]

# Calculate the difference in computer between US and China
difference = us_data["Computer"] - china_data["Computer"]

# Plot the difference over time
plt.plot(data["Time"], difference)
plt.xlabel("Time")
plt.ylabel("Difference in Computer")
plt.title("Difference in Computer between US and China over Time")
plt.show()
