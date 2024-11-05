import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the CSV file
file_path = './JSON/udp_data.csv'  # 请将路径替换为你的文件路径
data = pd.read_csv(file_path, header=None)

# Define the step size and the window size
step_size = 5
window_size = 64

# Create a directory to save the images
output_dir = './image/udp_data_images'
os.makedirs(output_dir, exist_ok=True)

# Function to create and save the images
def create_images(data, window_size, step_size, output_dir):
    num_rows = data.shape[0]
    img_count = 0

    for start in range(0, num_rows - window_size + 1, step_size):
        window_data = data.iloc[start:start + window_size].values
        img_matrix = np.array(window_data)

        # Apply the highlight condition
        highlight = img_matrix > 20
        cmap = plt.cm.gray

        plt.figure(figsize=(6, 6))
        plt.imshow(img_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=255)
        plt.colorbar()
        plt.title(f'32x32 Matrix starting at row {start}')
        plt.axis('off')  # Hide the axis
        plt.savefig(f'{output_dir}/img_{img_count}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        img_count += 1

# Generate the images
create_images(data, window_size, step_size, output_dir)
           
