from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the .h5 model
model = load_model('/home/lenovo/Downloads/test_model_1.h5')

# Path to your test image
# test_image_path = '/home/lenovo/mnt/hackathon/Detection-of-Sensitive-Data-Exposure-in-Images/images/images.png'

# Path to your folder containing images
folder_path = '/home/lenovo/mnt/hackathon/Detection-of-Sensitive-Data-Exposure-in-Images/output/images'

# Initialize an empty dictionary to store file name -> img_array mapping
images_dict = {}

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Filter for image files
        # Load and preprocess the image
        img_path = os.path.join(folder_path, filename)
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.  # Normalizing pixel values
        
        # Store the processed image array in the dictionary with filename as key
        images_dict[filename] = img_array

from datetime import datetime

# Get the current date and time
start_time = datetime.now()

# Check the dictionary with file name -> img_array mapping
for filename, img_array in images_dict.items():
    # print(f"File: {filename}, Image Array Shape: {img_array.shape}")
    prediction = model.predict(img_array)
    # Process the predictions or use them as needed

    # Print the prediction result
    if prediction[0] > 0.5:
        print(f"{filename} Prediction: Sensitive")
    # else:
    #     print("NONONO")

# Calculate the difference between the datetimes
time_difference = datetime.now() - start_time

# Calculate the difference in seconds
difference_in_seconds = time_difference.total_seconds()
print(difference_in_seconds)
