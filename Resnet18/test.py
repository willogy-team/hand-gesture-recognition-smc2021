import argparse
import json

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


parser = argparse.ArgumentParser()
parser.add_argument('-mp', '--model_path', type=str, default="./models",
                    help='Path to trained model')
parser.add_argument('-ncl', '--num_classes', type=int, default=2,
                    help="number of classes")
parser.add_argument('-td', '--test_dir', default="/media/data/dataset/hand_wash_mov_cls",
                    help="test directory path, include multi-folder, with each folder is a class")
parser.add_argument('-bz', '--batch_size', type=int, default=1,
                    help="batch size")
parser.add_argument('-iz', '--image_size', type=int, default=224,
                    help="image size for test")
args = parser.parse_args()


data_transforms = transforms.Compose([
                    transforms.Resize(args.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
image_dataset = datasets.ImageFolder(args.test_dir, data_transforms)
test_dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)


checkpoint = torch.load(args.model_path)
class_to_idx = checkpoint['class_to_idx']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, args.num_classes)
model.to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

groundtruths = []
predictions = []
with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        groundtruths.extend(labels.tolist())
        predictions.extend(predicted.tolist())

trainOn = args.model_path.split('/')[-2][5:]
testOn = args.test_dir.split('/')[-1][5:]
with open(''.join(['./output/', 'train', trainOn, '_test', testOn, '.json']), 'w') as f:
    json_dict = {}
    json_dict['y_true'] = groundtruths
    json_dict['y_pred'] = predictions
    json.dump(json_dict, f)

classes = [key for key, val in class_to_idx.items()]
print('[+] class_to_idx: ', class_to_idx)
print('[+] groundtruths: ', set(groundtruths))
print('[+] predictions: ', set(predictions))
print("[INFO] Acc: ", accuracy_score(groundtruths, predictions))
print("[INFO] Macro: ", f1_score(groundtruths, predictions, average='macro'))
print("[INFO] Micro: ", f1_score(groundtruths, predictions, average='micro'))
print("[INFO] For each class: \n", classification_report(groundtruths, predictions, target_names=classes, digits=4))
