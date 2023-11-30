from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the .h5 model
model = load_model('/home/lenovo/mnt/hackathon/Detection-of-Sensitive-Data-Exposure-in-Images/models/image_model.h5')

# Path to your test image
test_image_path = '/home/lenovo/mnt/hackathon/Detection-of-Sensitive-Data-Exposure-in-Images/images/images.png'

# Load and preprocess the test image
img = image.load_img(test_image_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array /= 255.  # Normalizing pixel values

# Make prediction using the loaded model
prediction = model.predict(img_array)
# Process the predictions or use them as needed
# print(predictions)

# Print the prediction result
if prediction[0] > 0.5:
    print("Prediction: Sensitive")
else:
    print("Prediction: Sensitive")