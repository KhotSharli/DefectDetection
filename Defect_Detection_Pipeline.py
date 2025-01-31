import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
from torchmetrics import Accuracy, F1Score, Precision, Recall

def process_coco_annotations(ann_path, image_dir):
    """Process COCO annotations to create binary labels"""
    coco = COCO(ann_path)
    image_ids = coco.getImgIds()
    images = coco.loadImgs(image_ids)
    
    data = []
    for img in images:
        ann_ids = coco.getAnnIds(imgIds=img['id'])
        annotations = coco.loadAnns(ann_ids)  # Fixed typo from loadAngs
        
        label = 0  # Default to no defect
        for ann in annotations:
            if ann['category_id'] in [1, 2, 3]:  # Defect categories
                label = 1
                break
            elif ann['category_id'] == 4:        # No defect
                label = 0
                break
                
        img_path = os.path.join(image_dir, img['file_name'])
        data.append({
            'image_path': img_path,
            'label': label
        })
    
    return pd.DataFrame(data)

class FoodDefectDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        label = self.df.iloc[idx]['label']
        img = Image.open(img_path).convert('RGB')
        return self.transform(img), torch.tensor(label, dtype=torch.float32)

def create_model(model_name='resnet50'):
    """Initialize specified model architecture"""
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif model_name == 'efficientnet':
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model

def train_model(model, train_loader, valid_loader, epochs=30, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    metrics = {
        'train_loss': [],
        'valid_loss': [],
        'valid_acc': [],
        'valid_f1': []
    }
    
    best_f1 = 0.0
    patience = 5 
    epochs_without_improvement = 0
    accuracy = Accuracy(task='binary').to(device)
    f1 = F1Score(task='binary').to(device)
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        # Validation
        model.eval()
        valid_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                
                preds = (torch.sigmoid(outputs) > 0.5).int()
                all_preds.append(preds)
                all_labels.append(labels.int())
        
        # Aggregate predictions
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        # Calculate metrics
        train_loss = running_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(valid_loader.dataset)
        valid_acc = accuracy(all_preds, all_labels).item()
        valid_f1 = f1(all_preds, all_labels).item()
        
        # Update metrics
        metrics['train_loss'].append(train_loss)
        metrics['valid_loss'].append(valid_loss)
        metrics['valid_acc'].append(valid_acc)
        metrics['valid_f1'].append(valid_f1)
        
        # Save the best model based on validation F1 score
        if valid_f1 > best_f1:
            best_f1 = valid_f1
            torch.save(model.state_dict(), f'best_{model.__class__.__name__}.pth')
            epochs_without_improvement = 0  # Reset counter
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch + 1} epochs")
            break
            
        # Update learning rate based on validation loss
        scheduler.step(valid_loss)
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}')
        print(f'Valid Acc: {valid_acc:.4f} | Valid F1: {valid_f1:.4f}')
        print('-'*50)
    
    return model, metrics


def evaluate_model(model, test_loader, model_name):
    """Model evaluation on test set"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(f'best_{model.__class__.__name__}.pth'))
    model.eval()
    
    criterion = nn.BCEWithLogitsLoss()
    test_loss = 0.0
    all_preds, all_labels = [], []
    
    metrics = {
        'accuracy': Accuracy(task='binary').to(device),
        'f1': F1Score(task='binary').to(device),
        'precision': Precision(task='binary').to(device),
        'recall': Recall(task='binary').to(device)
    }
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f'Evaluating {model_name}'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            test_loss += criterion(outputs, labels).item() * inputs.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.append(preds)
            all_labels.append(labels)
    
    results = {
        'test_loss': test_loss / len(test_loader.dataset),
        'accuracy': metrics['accuracy'](torch.cat(all_preds), torch.cat(all_labels)).cpu().item(),
        'f1': metrics['f1'](torch.cat(all_preds), torch.cat(all_labels)).cpu().item(),
        'precision': metrics['precision'](torch.cat(all_preds), torch.cat(all_labels)).cpu().item(),
        'recall': metrics['recall'](torch.cat(all_preds), torch.cat(all_labels)).cpu().item()
    }
    
    print(f'\n{model_name} Results:')
    print(f'Test Loss: {results["test_loss"]:.4f}')
    print(f'Accuracy: {results["accuracy"]:.4f} | F1 Score: {results["f1"]:.4f}')
    print(f'Precision: {results["precision"]:.4f} | Recall: {results["recall"]:.4f}')
    
    return results

def predict_defect(image_path, model_path, model_arch='resnet50'):
    """Make prediction on a single image"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = create_model(model_arch)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device).eval()
    
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()
    
    return ('Defect', prob) if prob > 0.5 else ('No Defect', prob)

def main(dataset_path):
    # Data preparation
    print("Processing dataset...")
    train_df = process_coco_annotations(
        os.path.join(dataset_path, "train/_annotations.coco.json"),
        os.path.join(dataset_path, "train")
    )
    valid_df = process_coco_annotations(
        os.path.join(dataset_path, "valid/_annotations.coco.json"),
        os.path.join(dataset_path, "valid")
    )
    test_df = process_coco_annotations(
        os.path.join(dataset_path, "test/_annotations.coco.json"),
        os.path.join(dataset_path, "test")
    )
    
    # Create datasets
    print("Creating data loaders...")
    train_dataset = FoodDefectDataset(train_df, transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))
    valid_dataset = FoodDefectDataset(valid_df)
    test_dataset = FoodDefectDataset(test_df)
    
    BATCH_SIZE = 32
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Train models
    print("\nTraining ResNet50...")
    resnet = create_model('resnet50')
    resnet, _ = train_model(resnet, train_loader, valid_loader)
    
    print("\nTraining EfficientNet...")
    effnet = create_model('efficientnet')
    effnet, _ = train_model(effnet, train_loader, valid_loader)
    
    # Evaluate models
    print("\nEvaluating models...")
    resnet_results = evaluate_model(resnet, test_loader, 'ResNet50')
    effnet_results = evaluate_model(effnet, test_loader, 'EfficientNet')
    
    # Save models
    torch.save(resnet.state_dict(), 'defect_detection_resnet50.pth')
    torch.save(effnet.state_dict(), 'defect_detection_efficientnet.pth')
    print("\nModels saved successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Defect Detection Pipeline')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to dataset directory containing train/valid/test folders')
    args = parser.parse_args()
    
    main(args.dataset_path)