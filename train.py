import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from tqdm import tqdm

from dataset import POC_Dataset 
from torch.utils.data import DataLoader, random_split

# Configuration
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 30
NUM_CLASSES = 4 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # add norm
])

# load dataset
full_dataset = POC_Dataset(data_dir='./POC_Dataset', data_type='Training', transform=transform)

dataset_size = len(full_dataset)
train_size = int(dataset_size * 0.8)
val_size = dataset_size - train_size

# valid spilit
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# load GoogLeNet
model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)

# loss, optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# Training start
print(f"Training... (Train: {len(train_dataset)}, Valid: {len(val_dataset)})")

best_acc = 0.0 

for epoch in range(NUM_EPOCHS):
    # Training
    model.train() 
    train_loss = 0.0
    
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        if isinstance(outputs, tuple):
            logits, aux2, aux1 = outputs
            loss = criterion(logits, labels) + 0.3 * criterion(aux1, labels) + 0.3 * criterion(aux2, labels)
        else:
            loss = criterion(outputs, labels)
            
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
    avg_train_loss = train_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Valid]"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct / total
    
    # print
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # save best
    if val_acc > best_acc:
        best_acc = val_acc
        save_path = "./googlenet_poc_best.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Best Model Saved (Acc: {best_acc:.2f}%)")

print(f"Training Finished")