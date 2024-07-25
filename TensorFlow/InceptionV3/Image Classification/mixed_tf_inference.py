import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Enable mixed precision
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Function to load the pretrained model
def load_pretrained_model():
    model = InceptionV3(weights='imagenet')
    return model

# Function to preprocess the image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to make predictions
def predict_image(model, image_path):
    img_array = preprocess_image(image_path)
    preds = model.predict(img_array)
    decoded_preds = decode_predictions(preds, top=1)[0][0]
    return decoded_preds

# Main function
if __name__ == "__main__":
    model = load_pretrained_model()

    image_path = 'cat.jpg'  # Path to the image you want to predict
    predicted_class = predict_image(model, image_path)
    print(f'The predicted class is: {predicted_class[1]} with a probability of {predicted_class[2]:.2f}')
