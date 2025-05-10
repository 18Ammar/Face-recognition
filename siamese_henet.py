# %%
import os
import random
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm
import matplotlib.pyplot as plt

# %%
# HEBlock 
class HEBlock(nn.Module):
    def __init__(self, beta=0.5):
        super(HEBlock, self).__init__()
        self.beta = beta

    def forward(self, x):
        B, C, H, W = x.size()
        heatmap = x.sum(dim=1, keepdim=True)
        heatmap_flat = heatmap.view(B, -1)
        k = int(self.beta * heatmap_flat.shape[1])
        _, top_indices = torch.topk(heatmap_flat, k=k, dim=1)
        mask = torch.ones_like(heatmap_flat)
        mask.scatter_(1, top_indices, 0)
        mask = mask.view(B, 1, H, W)
        return x * mask

# %%
class HENetEmbed(nn.Module):
    def __init__(self, feature_dim=256, beta=0.5):
        super(HENetEmbed, self).__init__()
        base = resnet18(pretrained=True)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.backbone = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )
        self.he_block = HEBlock(beta)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base.fc.in_features, feature_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.backbone(x)
        x = self.he_block(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return self.dropout(x)

# %%
class SiameseFontDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_fonts=100):
        self.root_dir = root_dir
        self.transform = transform

        all_classes = os.listdir(root_dir)
        self.classes = random.sample(all_classes, min(max_fonts, len(all_classes)))

        self.class_to_images = {
            c: [os.path.join(root_dir, c, f) for f in os.listdir(os.path.join(root_dir, c))]
            for c in self.classes
        }

        self.image_pairs = []
        for c in self.classes:
            images = self.class_to_images[c]
            for img in images:
                if len(images) < 2:
                    continue
                other = random.choice([x for x in images if x != img])
                self.image_pairs.append((img, other, 1))

                neg_class = random.choice([x for x in self.classes if x != c])
                neg_img = random.choice(self.class_to_images[neg_class])
                self.image_pairs.append((img, neg_img, 0))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        path1, path2, label = self.image_pairs[idx]
        img1 = Image.open(path1).convert("L")
        img2 = Image.open(path2).convert("L")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)

# %%
def contrastive_loss(x1, x2, label, margin=1.0):
    euclidean_distance = (x1 - x2).pow(2).sum(1).sqrt()
    loss = label * euclidean_distance.pow(2) + (1 - label) * torch.clamp(margin - euclidean_distance, min=0).pow(2)
    return loss.mean()

# %%
transform = transforms.Compose([
    transforms.Resize((64, 224)),
    transforms.ToTensor(),
    transforms.RandomApply([
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
        transforms.RandomResizedCrop((64, 224), scale=(0.85, 1.0)),
        transforms.GaussianBlur(3),
        transforms.RandomErasing()
    ], p=0.7),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

dataset = SiameseFontDataset(
    "D:\\Harf\\splited_datasets-20250425T042643Z-001\\splited_datasets\\train",
    transform=transform,
    max_fonts=100
)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# %%
model = HENetEmbed().cuda()
model.load_state_dict(torch.load("resnet18_henet_siamese_fonts_epoch5.pth"))
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# %%
num_epochs = 5
losses = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for img1, img2, label in tqdm(loader, desc=f"Epoch {epoch+1}"):
        img1, img2, label = img1.cuda(), img2.cuda(), label.cuda()
        feat1 = model(img1)
        feat2 = model(img2)
        loss = contrastive_loss(feat1, feat2, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    losses.append(avg_loss)
    scheduler.step()

    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f"henet_siamese_epoch{epoch+1}.pth")

# %%
# Plot the training loss curve
plt.plot(range(1, num_epochs+1), losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.grid()
plt.show()

# %%
torch.save(model.state_dict(), "resnet18_henet_siamese_fonts_epoch5.pth")


# %%
def evaluate(model, dataloader, threshold=0.5):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for img1, img2, label in tqdm(dataloader, desc="Evaluating"):
            img1, img2, label = img1.cuda(), img2.cuda(), label.cuda()
            feat1 = model(img1)
            feat2 = model(img2)
            distance = (feat1 - feat2).pow(2).sum(1).sqrt()

            predicted = (distance < threshold).float()
            correct += (predicted == label).sum().item()
            total += label.size(0)

    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy

# %%
val_dataset = SiameseFontDataset(
    "D:\\Harf\\splited_datasets-20250425T042643Z-001\\splited_datasets\\test",  
    transform=transform,
    max_fonts=100
)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

evaluate(model, val_loader, threshold=0.6)

# %%
