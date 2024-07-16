import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor
from omegaconf import OmegaConf
from tqdm import tqdm
from PIL import Image
from src.models11 import ImageEncoder
from src.datasets10 import ImageDataset
from src.loss import ClipLoss

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="dinov2")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")

class PretrainModel(nn.Module):
    def __init__(self):
        super(PretrainModel, self).__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        for param in self.clip_model.parameters():
            param.requires_grad = False
        for param in self.clip_model.vision_model.parameters():
            param.requires_grad = True

    def forward(self, images):
        image_features = self.clip_model.get_image_features(images)
        return image_features

def contrastive_loss(features, temperature=0.07):
    features = F.normalize(features, dim=-1)
    logits = torch.matmul(features, features.t()) / temperature
    labels = torch.arange(features.size(0), device=features.device)
    loss = F.cross_entropy(logits, labels)
    return loss

def pretrain_image_encoder(model, image_loader, device, epochs=9, lr=1e-5, accumulation_steps=2):
    optimizer = optim.AdamW(model.clip_model.vision_model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    clip_loss_fn = ClipLoss()
    mse_loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        print(f"Pretraining Epoch {epoch+1}/{epochs}")
        epoch_loss = 0
        optimizer.zero_grad()
        for step, images in enumerate(tqdm(image_loader, desc="Pretraining")):
            images = images.to(device)

            image_features = model(images)

            loss = contrastive_loss(image_features) / accumulation_steps
            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * accumulation_steps

        scheduler.step()
        print(f"Pretraining Epoch {epoch+1}/{epochs} | Loss: {epoch_loss / len(image_loader)}")

def extract_image_features(model, image_loader, device):
    model.eval()
    features = []
    with torch.no_grad():
        for images in tqdm(image_loader, desc="Extracting image features"):
            images = images.to(device)
            image_features = model(images)
            features.append(image_features.cpu())
    features = torch.cat(features, dim=0)
    return features

def main():
    config = OmegaConf.load('configs/config.yaml')
    train_image_dataset = ImageDataset(
        image_paths_file=os.path.join(config.data_dir, "train_image_paths.txt"),
        base_dir=config.image_dir
    )
    val_image_dataset = ImageDataset(
        image_paths_file=os.path.join(config.data_dir, "val_image_paths.txt"),
        base_dir=config.image_dir
    )

    train_image_loader = DataLoader(train_image_dataset, config.pre_batch_size, shuffle=True, num_workers=config.num_workers)
    val_image_loader = DataLoader(val_image_dataset, config.pre_batch_size, shuffle=False, num_workers=config.num_workers)

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = PretrainModel().to(config.device)
    pretrain_image_encoder(model, train_image_loader, config.device, epochs=config.image_pretrain_epochs, lr=1e-4, accumulation_steps=4)

    torch.save(model.state_dict(), "data/pretrained_model.pth")

    image_dataset = ImageDataset(
        image_paths_file=os.path.join(config.data_dir, "train_image_paths.txt"),
        base_dir=config.image_dir
    )
    image_loader = DataLoader(image_dataset, config.pre_batch_size, shuffle=False, num_workers=config.num_workers)
    model = PretrainModel().to(config.device)
    model.load_state_dict(torch.load("data/pretrained_model.pth"))
    image_features = extract_image_features(model, image_loader, config.device)
    torch.save(image_features, "data/image_features.pt")

if __name__ == "__main__":
    main()
