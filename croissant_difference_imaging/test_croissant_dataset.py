import mlcroissant as mlc
import numpy as np
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 5

class DifferenceImagingDataset(Dataset):
    def __init__(self, json_path, root_dir, crop_size=64):
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.half_size = crop_size // 2
        
        try:
            self.ds = mlc.Dataset(jsonld=json_path)
            self.records = list(self.ds.records("transient_candidates"))
            print(f"Loaded {len(self.records)} records.")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.records = []

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        
        label = record.get("transient_candidates/label")
        x = record.get("transient_candidates/x")
        y = record.get("transient_candidates/y")
        rel_path = record.get("transient_candidates/image_path")
        
        if isinstance(rel_path, bytes):
            rel_path = rel_path.decode('utf-8')
            
        full_path = os.path.join(self.root_dir, rel_path)
        
        full_image = np.load(full_path, mmap_mode='r') 
        
        x, y = int(x), int(y)
        
        y_min = max(0, y - self.half_size)
        y_max = min(full_image.shape[0], y + self.half_size)
        x_min = max(0, x - self.half_size)
        x_max = min(full_image.shape[1], x + self.half_size)
        
        cutout = full_image[y_min:y_max, x_min:x_max, :]
        
        # padding if cutout is smaller than 64x64
        if cutout.shape[0] != self.crop_size or cutout.shape[1] != self.crop_size:
            pad_h = self.crop_size - cutout.shape[0]
            pad_w = self.crop_size - cutout.shape[1]
            cutout = np.pad(cutout, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')

        tensor = torch.from_numpy(cutout).permute(2, 0, 1).float()
        
        features = []
        for feat in ['sharpness', 'roundness1', 'roundness2', 'npix', 'peak', 'flux', 'mag', 'daofind_mag']:
            val = record.get(f"transient_candidates/{feat}")
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                features.append(float(val))
            else:
                features.append(0.0)
        
        features_tensor = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label).float().unsqueeze(0)

        return tensor, features_tensor, label

class DiffImagingCNN(nn.Module):
    def __init__(self, num_features=8):
        super(DiffImagingCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(7, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8 + num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, features):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = torch.cat([x, features], dim=1)
        x = self.classifier(x)
        return x

def train_model(dataset_dir):
    json_path = os.path.join(dataset_dir, "croissant.json")
    dataset = DifferenceImagingDataset(json_path, dataset_dir)
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")
    
    model = DiffImagingCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, features, labels) in enumerate(loader):
            inputs, features, labels = inputs.to(device), features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs, features)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            if i % 5 == 0:
                print(f"  [Epoch {epoch+1}, Batch {i}] Loss: {loss.item():.4f}")

        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch+1} Complete. Avg Loss: {running_loss/len(loader):.4f} | Acc: {epoch_acc:.2f}%")

    print("\nTraining completed")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test difference imaging dataset with CNN training")
    parser.add_argument("--dataset_dir", "-d", type=str, default="./hackathon_dataset",
                        help="Dataset directory containing croissant.json (default: ./hackathon_dataset)")
    args = parser.parse_args()
    
    train_model(args.dataset_dir)
