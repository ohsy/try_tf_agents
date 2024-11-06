
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Read the contents of the CSV file
# Replace 'data.csv' with the path to your CSV file
file_path = 'data.csv'
data = pd.read_csv(file_path)

# Step 2: Plot the data
plt.figure(figsize=(10, 6))
plt.plot(data['train_step'], data['avg_return'], marker='o', linestyle='-', color='b')
plt.title('Average Return vs Train Step')
plt.xlabel('Train Step')
plt.ylabel('Average Return')
plt.grid()
plt.xticks(rotation=45)

# Step 3: Save the plot to a file
# You can specify the file format by changing the extension (e.g., .png, .jpg, .pdf)
output_file = 'avg_return_plot.png'
plt.savefig(output_file)

# Optionally, show the plot
plt.show()

print(f"Plot saved as {output_file}")

