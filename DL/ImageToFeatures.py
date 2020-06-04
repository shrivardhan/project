import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from PIL import Image
from torchvision import datasets, models, transforms
import os
import copy
import json
import subprocess

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

model = models.inception_v3(pretrained=True)
model.eval()
model.fc = Identity()

preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image_path = "/Volumes/Seagate Bac/VIST_Dataset/COCOImages/"

feature_path = "./Dataset/COCOFeatures/"

coco_desc_path = "./Dataset/coco_train_captions.json"

for image_file in os.listdir(image_path):
    featurename = str(int(image_file[15:-4]))
    if image_file.endswith(".jpg"):
        print(image_file)
        try:
            if os.path.exists(feature_path+featurename):
                continue
            image = Image.open(image_path+image_file)
            image = image.convert("RGB")
            input_tensor = preprocess(image)
            if input_tensor.shape!=(3,299,299):
                continue
            input_batch = input_tensor.unsqueeze(0)
            output = model(input_batch)
            output = output.reshape(output.shape[1],1)
            torch.save(output,feature_path+featurename)
        except:
            cmd = ["rm",image_path+image_file]
            subprocess.call(cmd)
            continue