import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

def get_data_loaders(root_dir, input_size, batch_size):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    train_ds = datasets.ImageFolder(os.path.join(root_dir, 'train'), transform)
    val_ds   = datasets.ImageFolder(os.path.join(root_dir, 'val'),   transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, len(train_ds.classes)

def build_model(num_classes, device):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

def train_and_validate(model, train_loader, val_loader, device, epochs, lr, patience, checkpoint_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for images, labels in tqdm(train_loader, desc=f"Train Epoch {epoch}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            running_corrects += (preds == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc  = running_corrects / len(train_loader.dataset)
        print(f"  Train Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Val   Epoch {epoch}/{epochs}"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_corrects += (preds == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc  = val_corrects / len(val_loader.dataset)
        print(f"  Val   Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # load best weights
    model.load_state_dict(torch.load(checkpoint_path))
    return model

data_root     = 'dataset'
input_size    = 224
batch_size    = 32
epochs        = 10
learning_rate = 1e-4
patience      = 3
checkpoint    = 'best_model.pth'
device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader, val_loader, num_classes = get_data_loaders(data_root, input_size, batch_size)
model = build_model(num_classes, device)
model = train_and_validate(model, train_loader, val_loader, device,
                           epochs, learning_rate, patience, checkpoint)
