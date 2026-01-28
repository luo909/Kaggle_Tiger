import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class Config:
    seed = 42
    img_size = 448

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(Config.seed)

train_transform = transforms.Compose(
    [
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.481, 0.457, 0.408], [0.268, 0.261, 0.275]),
        transforms.RandomErasing(p=0.25),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.481, 0.457, 0.408], [0.268, 0.261, 0.275]),
    ]
)

def denormalize(tensor):
    mean = torch.tensor([0.481, 0.457, 0.408]).view(3, 1, 1)
    std = torch.tensor([0.268, 0.261, 0.275]).view(3, 1, 1)
    return tensor * std + mean

def tensor_to_image(tensor):
    img = denormalize(tensor)
    img = torch.clamp(img, 0, 1)
    img = img.permute(1, 2, 0).numpy()
    return img

train_df = pd.read_csv("jaguar-re-id/train.csv")
train_dir = Path("jaguar-re-id/train/train")

img1_name = train_df.iloc[0]["filename"]
img2_name = train_df.iloc[10]["filename"]

img1_path = train_dir / img1_name
img2_path = train_dir / img2_name

img1 = Image.open(img1_path).convert("RGB")
img2 = Image.open(img2_path).convert("RGB")

fig, axes = plt.subplots(2, 6, figsize=(20, 7))

axes[0, 0].imshow(img1)
axes[0, 0].set_title(f"Original: {img1_name}", fontsize=12)
axes[0, 0].axis('off')

axes[1, 0].imshow(img2)
axes[1, 0].set_title(f"Original: {img2_name}", fontsize=12)
axes[1, 0].axis('off')

for i in range(1, 6):
    transformed1 = train_transform(img1)
    img1_display = tensor_to_image(transformed1)
    axes[0, i].imshow(img1_display)
    axes[0, i].set_title(f"Transform {i}", fontsize=12)
    axes[0, i].axis('off')

    transformed2 = train_transform(img2)
    img2_display = tensor_to_image(transformed2)
    axes[1, i].imshow(img2_display)
    axes[1, i].set_title(f"Transform {i}", fontsize=12)
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('transform_visualization.png', dpi=150, bbox_inches='tight')
print("可视化结果已保存到 transform_visualization.png")
plt.show()
