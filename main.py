import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from src.datasets24 import ThingsMEGDataset, ImageDataset
from src.models27 import MEGClassifier
from src.loss import ClipLoss
from src.utils import set_seed

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    
    # MEGデータの読み込み
    train_set = ThingsMEGDataset("train", args.data_dir)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    
    val_set = ThingsMEGDataset("val", args.data_dir)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    
    test_set = ThingsMEGDataset("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # 事前に抽出した画像特徴をロード
    image_features = torch.load(os.path.join(args.data_dir, "image_features.pt")).to(args.device)
    
    # ------------------
    #       Model
    # ------------------
    num_subjects = len(torch.unique(train_set.subject_idxs))
    model = MEGClassifier(
        num_classes=train_set.num_classes, 
        num_subjects=num_subjects, 
        in_channels=args.in_channels, 
        out_channels=args.out_channels, 
        depth=args.depth, 
        dilation=args.dilation
    ).to(args.device)

    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-5)
    scheduler1 = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # ------------------
    #   Loss Functions
    # ------------------  
    clip_loss_fn = ClipLoss()
    mse_loss_fn = nn.MSELoss()

    # ------------------
    #   Start training
    # ------------------  
    max_val_acc = 0
    accuracy = Accuracy(task="multiclass", num_classes=train_set.num_classes, top_k=10).to(args.device)


    accumulation_steps = args.accumulation_steps
    early_stop_count = 0
    early_stop_patience = args.early_stop_patience
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        train_loss, train_acc, val_loss, val_acc = [], [], [], []

        model.train()
        #optimizer.zero_grad()
        for i, (meg_batch) in enumerate(tqdm(train_loader, desc="Train")):
            X, y, subject_idxs = meg_batch

            X, y, subject_idxs = X.to(args.device), y.to(args.device), subject_idxs.to(args.device)
            image_X = image_features[subject_idxs]

            y_pred, meg_output, img_output = model(X, subject_idxs, image_X)

            loss = F.cross_entropy(y_pred, y)
            loss = loss / args.accumulation_steps
            loss.backward()
                        
            if (i + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss.append(loss.item() * args.accumulation_steps)
            train_acc.append(accuracy(y_pred, y).item())

        model.eval()
        for meg_batch in tqdm(val_loader, desc="Validation"):
            X, y, subject_idxs = meg_batch

            X, y, subject_idxs = X.to(args.device), y.to(args.device), subject_idxs.to(args.device)
            image_X = image_features[subject_idxs]

            with torch.no_grad():
                y_pred, meg_output, img_output = model(X, subject_idxs, image_X)

            loss = F.cross_entropy(y_pred, y)
            val_loss.append(loss.item())
            val_acc.append(accuracy(y_pred, y).item())

        scheduler1.step(np.mean(val_loss))

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = np.mean(val_acc)
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= early_stop_patience:
            print("Early stopping")
            break

    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

    preds = []
    model.eval()
    for meg_batch in tqdm(test_loader, desc="Validation"):
        X, subject_idxs = meg_batch
        image_X = image_features[subject_idxs]

        with torch.no_grad():
            y_pred, meg_output, img_output = model(X.to(args.device), subject_idxs.to(args.device), image_X)
        preds.append(y_pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")

if __name__ == "__main__":
    run()
