import numpy as np
import matplotlib.pyplot as plt

# Load the multiband npy file
file_path = r"Data\\Pre-Processed\\Jaipur\\2013-04-19.npy"
data = np.load(file_path)

# Check the shape and data type of the loaded data
print(f'Shape of the data: {data.shape}')
print(f'Data type: {data.dtype}')

# Visualize each band separately
num_bands = data.shape[0]

for i in range(num_bands):
    plt.figure(figsize=(6, 6))
    plt.imshow(data[i], cmap='gray')
    plt.title(f'Band {i + 1}')
    plt.axis('off')  # Hide axis
    plt.show()
