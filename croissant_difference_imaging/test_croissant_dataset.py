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
    def __init__(self, json_path, root_dir, crop_size=64, use_algorithm='both'):
        """
        Args:
            use_algorithm: 'zogy', 'sfft', or 'both' - which features to use
        """
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.half_size = crop_size // 2
        self.use_algorithm = use_algorithm
        
        try:
            self.ds = mlc.Dataset(jsonld=json_path)
            self.records = list(self.ds.records("transient_candidates"))
            print(f"Loaded {len(self.records)} transient candidates.")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.records = []

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        prefix = "transient_candidates"
        
        label = record.get(f"{prefix}/label")
        x = record.get(f"{prefix}/x")
        y = record.get(f"{prefix}/y")
        rel_path = record.get(f"{prefix}/image_path")
        
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
        
        # Collect features based on algorithm choice
        features = []
        feature_names = ['sharpness', 'roundness1', 'roundness2', 'npix', 'peak', 'flux', 'mag', 'daofind_mag', 'flags']
        
        if self.use_algorithm in ['zogy', 'both']:
            for feat in feature_names:
                val = record.get(f"{prefix}/zogy_{feat}")
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    features.append(float(val))
                else:
                    features.append(0.0)
        
        if self.use_algorithm in ['sfft', 'both']:
            for feat in feature_names:
                val = record.get(f"{prefix}/sfft_{feat}")
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    features.append(float(val))
                else:
                    features.append(0.0)
        
        features_tensor = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label).float().unsqueeze(0)

        return tensor, features_tensor, label

class DiffImagingCNN(nn.Module):
    """Simple 3-layer CNN for transient classification"""
    def __init__(self, num_features=18):
        super(DiffImagingCNN, self).__init__()
        
        # 3-layer CNN for image processing
        self.conv1 = nn.Conv2d(9, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Calculate flattened size: 64x64 -> 32x32 -> 16x16 -> 8x8
        # 8x8x128 = 8192
        self.fc1 = nn.Linear(8192 + num_features, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x, features):
        # 3 conv layers with pooling
        x = self.pool(self.relu(self.conv1(x)))  # 64x64 -> 32x32
        x = self.pool(self.relu(self.conv2(x)))  # 32x32 -> 16x16
        x = self.pool(self.relu(self.conv3(x)))  # 16x16 -> 8x8
        
        # Flatten and concatenate with features
        x = torch.flatten(x, 1)
        x = torch.cat([x, features], dim=1)
        
        # Fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

def test_dataset_loading(dataset_dir):
    """Quick dataset validation and statistics"""
    json_path = os.path.join(dataset_dir, "croissant.json")
    
    print("=" * 60)
    print("Dataset Loading & Statistics")
    print("=" * 60)
    
    dataset = DifferenceImagingDataset(json_path, dataset_dir, use_algorithm='both')
    
    if len(dataset) == 0:
        print("ERROR: No records found")
        return
    
    print(f"\nLoaded {len(dataset)} candidates")
    
    # Sample data point
    img, feats, lbl = dataset[0]
    print(f"Image tensor: {img.shape}")
    print(f"Features: {feats.shape} (9 ZOGY + 9 SFFT)")
    print(f"Labels: binary classification")
    
    # Quick stats
    prefix = "transient_candidates"
    labels = [r.get(f"{prefix}/label") for r in dataset.records]
    real_count = sum(1 for lbl in labels if lbl == 1)
    bogus_count = sum(1 for lbl in labels if lbl == 0)
    
    injected = [r.get(f"{prefix}/injected") for r in dataset.records]
    injected_count = sum(1 for val in injected if val is True or val == 'True' or val == 1)
    
    print(f"\nReal: {real_count}, Bogus: {bogus_count}")
    print(f"Injected sources: {injected_count}")
    print(f"Class balance: 1:{bogus_count/max(real_count,1):.1f}")
    
    print("\n" + "=" * 60)

def train_model(dataset_dir, algorithm="both"):
    """Train a model using specified algorithm features
    Args:
        algorithm: 'zogy', 'sfft', or 'both' - which features to use
    """
    json_path = os.path.join(dataset_dir, "croissant.json")
    dataset = DifferenceImagingDataset(json_path, dataset_dir, use_algorithm=algorithm)
    
    if len(dataset) == 0:
        print(f"No data found. Skipping training.")
        return None
    
    # Determine number of features based on algorithm
    if algorithm == 'both':
        num_features = 18  # 9 ZOGY + 9 SFFT features
    else:
        num_features = 9  # Either ZOGY or SFFT features
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on {device} using {algorithm.upper()} features...")
    print(f"Feature count: {num_features}")
    
    model = DiffImagingCNN(num_features=num_features).to(device)
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

    print(f"\nTraining with {algorithm.upper()} features completed")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test unified difference imaging dataset with CNN training")
    parser.add_argument("--dataset_dir", "-d", type=str, default="./hackathon_dataset",
                        help="Dataset directory containing croissant.json (default: ./hackathon_dataset)")
    parser.add_argument("--algorithm", "-a", type=str, default="both", choices=["zogy", "sfft", "both"],
                        help="Which algorithm features to use: zogy, sfft, or both (default: both)")
    args = parser.parse_args()
    
    # Test loading and analyze dataset
    test_dataset_loading(args.dataset_dir)
    
    # Train model
    print("\n" + "=" * 60)
    print(f"Training Model with {args.algorithm.upper()} Features")
    print("=" * 60)
    
    model = train_model(args.dataset_dir, args.algorithm)
