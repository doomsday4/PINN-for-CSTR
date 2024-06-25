import pandas as pd
import matplotlib.pyplot as plt

# Load the data
# Replace 'cstr_data.csv' with the path to your CSV file
data = pd.read_csv('CSTR_jacket_simData.csv')

# Display the first few rows of the dataframe
print(data.head())

# Plot Tc_out against Time
plt.figure(figsize=(10, 6))
plt.plot(data['Time'], data['Tc_out'], marker='o', linestyle='-', color='b')
plt.title('Time vs Tc_out')
plt.xlabel('Time')
plt.ylabel('Tc_out')
plt.grid(True)
plt.show()