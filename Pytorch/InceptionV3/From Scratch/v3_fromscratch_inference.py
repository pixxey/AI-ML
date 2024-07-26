import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from torchvision import models

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the InceptionV3 model from scratch
class InceptionV3Scratch(nn.Module):
    def __init__(self, num_classes=1000):
        super(InceptionV3Scratch, self).__init__()
        self.model = models.inception_v3(pretrained=False, aux_logits=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Function to load the built-from-scratch model
def load_built_from_scratch_model(model_path, num_classes):
    model = InceptionV3Scratch(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model

# Function to preprocess the image
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Function to load class names
def load_class_names(filepath):
    with open(filepath) as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

# Function to make predictions
def predict_image(model, image_path, class_names):
    image = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        predicted_class = class_names[preds[0]]
    return predicted_class

# Main function
if __name__ == "__main__":
    # Load the built-from-scratch model
    num_classes = 120  # Number of classes in the Stanford Dogs dataset
    model_path = 'inceptionv3_scratch_stanford_dogs.pth'  # Path to the saved model
    model = load_built_from_scratch_model(model_path, num_classes)
    
    # Load class names from a file
    class_names = load_class_names('imagenet_classes.txt')  # Replace with the path to your class names file

    # Path to the image you want to predict
    image_path = 'dog.jpg'  # Replace with the path to your image
    predicted_class = predict_image(model, image_path, class_names)
    print(f'The predicted class is: {predicted_class}')
