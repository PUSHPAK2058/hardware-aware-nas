# Baseline Face Recognition Model Training

This project contains a baseline model training pipeline for face recognition dataset using PyTorch.

## Requirements

- Python 3.7+
- PyTorch
- torchvision

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
```

2. Activate the virtual environment:

- Windows:
  ```bash
  .\venv\Scripts\activate
  ```
- Linux/macOS:
  ```bash
  source venv/bin/activate
  ```

3. Install required packages:

```bash
pip install torch torchvision
```

## Usage

Prepare your face recognition dataset in ImageFolder format with subdirectories per class/person:

```
dataset/
├── person1/
│   ├── img1.jpg
│   ├── img2.jpg
├── person2/
│   ├── img1.jpg
│   ├── img2.jpg
...
```

Run the training script:

```bash
python train_baseline.py --data_dir path/to/dataset --batch_size 32 --epochs 20 --lr 0.001 --checkpoint_path model.pth
```

The best model checkpoint will be saved at the specified path.

## Notes

- This is a baseline model using ResNet18 pretrained on ImageNet for face recognition classification.
- You can modify `train_baseline.py` to customize data augmentation, model architecture, or loss functions.
