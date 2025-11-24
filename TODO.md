# TODO for Baseline Face Recognition Model Training

## Completed
- Created `train_baseline.py`:
  - Loads and preprocesses face recognition dataset in ImageFolder format.
  - Defines ResNet18 pretrained backbone with classification head.
  - Implements training loop with cross-entropy loss.
  - Implements evaluation on validation split.
  - Saves best model checkpoint based on validation accuracy.

- Created `README.md`:
  - Instructions for environment setup.
  - Instructions for dataset preparation.
  - Instructions for running the training script.

## Next Steps
- User to prepare dataset in ImageFolder format.
- User to set up virtual environment and install PyTorch, torchvision dependencies.
- Run training using `python train_baseline.py --data_dir path/to/dataset`.
- Optionally, extend model architecture, data augmentation, or implement advanced loss functions.

## Future Enhancements
- Support for triplet or contrastive loss for embedding learning.
- Integration of more sophisticated face recognition datasets.
- Model evaluation using standard face verification metrics.
- Add logging and checkpoint resume capabilities.
