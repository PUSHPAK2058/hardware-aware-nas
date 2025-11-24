import os
import shutil
import torch
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from train_baseline import create_model, train_one_epoch, evaluate

def create_fake_dataset(num_classes=2, num_samples=20, image_size=(3, 224, 224)):
    # Use torchvision FakeData dataset as a minimal test dataset
    dataset = FakeData(size=num_samples, image_size=image_size, num_classes=num_classes, transform=ToTensor())
    return dataset

def test_training_loop():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = 2
    dataset = create_fake_dataset(num_classes=num_classes, num_samples=40)
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = create_model(num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    initial_val_loss, initial_val_acc = evaluate(model, device, val_loader, criterion)
    print(f"Initial validation loss: {initial_val_loss:.4f}, acc: {initial_val_acc:.4f}")

    # train one epoch
    train_loss, train_acc = train_one_epoch(model, device, train_loader, criterion, optimizer)
    print(f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")

    # validate again
    val_loss, val_acc = evaluate(model, device, val_loader, criterion)
    print(f"Validation loss after 1 epoch: {val_loss:.4f}, acc: {val_acc:.4f}")

    # Check if loss decreased or accuracy increased after training
    assert val_loss <= initial_val_loss or val_acc >= initial_val_acc, "Validation performance did not improve after training."

if __name__ == '__main__':
    test_training_loop()
    print("Critical-path test passed.")
