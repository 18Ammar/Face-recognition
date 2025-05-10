# HENet-Siamese Face Recognition Network

This project implements a face recognition system using a Siamese neural network architecture based on ResNet18, enhanced with a custom `HEBlock` for attention-based feature suppression. It is trained using contrastive loss on pairs of face images to learn embeddings that reflect similarity.

## Features

- ResNet18-based Siamese architecture
- HEBlock to suppress uninformative regions in the feature map
- Contrastive loss for learning similarity between image pairs
- Custom dataset loader for face pairs generation
- Visualization of training loss
- Evaluation using Euclidean distance threshold

## Dataset Structure

The dataset should be organized as follows:

```
face_datasets/
├── person1/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── person2/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
└── ...
```

Each subdirectory should contain face images of a single person.

## Dependencies

- Python 3.x
- PyTorch
- torchvision
- Pillow
- matplotlib
- tqdm

Install them via pip:

```bash
pip install torch torchvision pillow matplotlib tqdm
```

## Training

1. Update the dataset path in the script.
2. Run the training cell to train the model.

```python
dataset = SiameseFaceDataset("path_to/face_datasets/train", transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## Evaluation

Evaluate the trained model using a threshold-based accuracy metric:

```python
val_dataset = SiameseFaceDataset("path_to/face_datasets/test", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
evaluate(model, val_loader, threshold=0.6)
```

## Output

- Training loss over epochs (visualized with matplotlib)
- Trained model weights saved every few epochs
- Final model saved as `resnet18_henet_siamese_faces_epoch5.pth`
- Evaluation accuracy on validation/test set


