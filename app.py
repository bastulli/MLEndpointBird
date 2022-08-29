import io
import json

from torch import nn
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
import torch

app = Flask(__name__)

imagenet_class_index = json.load(open('data.json'))
path = 'model/model.pt'
AlexNet_model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
AlexNet_model.classifier[4] = nn.Linear(4096,1024)
AlexNet_model.classifier[6] = nn.Linear(1024,len(imagenet_class_index))
device = torch.device("cpu")
AlexNet_model.to(device)
model = AlexNet_model
#model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
CheckPoint = torch.load(path)
model.load_state_dict(CheckPoint['model_state_dict'])
model.eval()

def transform_image(image_bytes):                                            
    my_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()])
        
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = torch.max(outputs, 1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_name': class_name})

if __name__ == '__main__':
    app.run()