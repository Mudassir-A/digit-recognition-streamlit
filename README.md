# MNIST Digit Recognizer

This project is a simple MNIST digit recognizer using TensorFlow and Streamlit. It includes scripts for training a model, generating sample images, and a web application for digit prediction.

## Project Structure

- `train.py`: Script to train the MNIST model
- `sample_images.py`: Script to generate sample MNIST images
- `app.py`: Streamlit web application for digit recognition
- `models/`: Directory to store the trained model
- `images/`: Directory to store generated sample images

## Setup

1. Install the required dependencies:
   ```bash
   pip install tensorflow numpy matplotlib pillow streamlit
   ```

2. Train the model:
   ```bash
   python train.py
   ```

3. (Optional) Generate sample images:
   ```bash
   python sample_images.py
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage

### Training the Model

Run `train.py` to train the MNIST model. The script will:
- Load and preprocess the MNIST dataset
- Create and compile a CNN model
- Train the model for 5 epochs
- Evaluate the model on the test set
- Save the trained model as `models/mnist_model.h5`

### Generating Sample Images

Run `sample_images.py` to generate sample MNIST images. By default, it will:
- Create 10 random sample images from the MNIST dataset
- Save the images in the `images/` directory

You can modify the number of samples by changing the `num_samples` parameter in the script.

### Using the Web Application

1. Ensure that the model has been trained and saved as `models/mnist_model.h5`.
2. Run the Streamlit app using `streamlit run app.py`.
3. Open the provided URL in your web browser.
4. Upload an image of a handwritten digit (0-9).
5. Click the "Predict" button to see the model's prediction.

## Note

This project is for educational purposes and demonstrates a basic implementation of digit recognition using the MNIST dataset. The model's accuracy may vary, and it's not intended for production use without further improvements and error handling.
