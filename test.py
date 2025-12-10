import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix 

from dataset import POC_Dataset

# Configuration
BATCH_SIZE = 64
NUM_CLASSES = 4
model_path = "./googlenet_poc_best.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# load dataset
test_dataset = POC_Dataset(data_dir='./POC_Dataset', data_type='Testing', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# load model
model = models.googlenet(aux_logits=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

try:
    model.load_state_dict(torch.load(model_path))
    print(f"loaded model: {model_path}")
except FileNotFoundError:
    print(f"Error {model_path})")
    exit()

model = model.to(device)

model.eval()

all_preds = []
all_labels = []

print("Evaluation")

# test start
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing"):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs) 
        
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# print
target_names = ['Chorionic_villi', 'Decidual_tissue', 'Hemorrhage', 'Trophoblastic_tissue']

print("Test Result")
# Precision, Recall, F1-score
print(classification_report(all_labels, all_preds, target_names=target_names))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))