import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

def create_mnist_samples(num_samples=10, output_dir='images'):
    # Load MNIST dataset
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Randomly select samples
    indices = np.random.choice(len(x_train), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        image = x_train[idx]
        label = y_train[idx]
        
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(3, 3))
        
        # Display the image
        ax.imshow(image, cmap='gray')
        ax.axis('off')
        
        # Save the figure
        filename = f'mnist_sample_{i}_label_{label}.png'
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
    
    print(f"{num_samples} MNIST samples saved in '{output_dir}' directory.")

# Example usage
if __name__ == "__main__":
    create_mnist_samples(num_samples=10)
