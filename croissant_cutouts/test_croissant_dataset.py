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


class RomanCutoutDataset(Dataset):
    def __init__(self, json_path, root_dir):
        self.root_dir = root_dir
        
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
        rel_path = record.get("transient_candidates/cutout_path")
        
        if isinstance(rel_path, bytes):
            rel_path = rel_path.decode('utf-8')
            
        full_path = os.path.join(self.root_dir, rel_path)
        
        # Load pre-extracted 64x64x4 cutout
        cutout = np.load(full_path)
        
        # Convert to tensor (C, H, W) format
        tensor = torch.from_numpy(cutout).permute(2, 0, 1).float()
        label = torch.tensor(label).float().unsqueeze(0)

        return tensor, label


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
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
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_model(dataset_dir):
    json_path = os.path.join(dataset_dir, "croissant.json")
    dataset = RomanCutoutDataset(json_path, dataset_dir)
    
    if len(dataset) == 0:
        print("No data loaded. Exiting.")
        return None
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")
    
    model = SimpleCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
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
    parser = argparse.ArgumentParser(description="Test and train on Croissant cutout dataset")
    parser.add_argument("--dataset_dir", "-d", type=str, default="./hackathon_dataset",
                        help="Dataset directory containing croissant.json (default: ./hackathon_dataset)")
    args = parser.parse_args()
    
    trained_model = train_model(args.dataset_dir)
