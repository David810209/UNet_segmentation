# Import all the packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
import os
from tqdm import tqdm
from unet_utils import UNet, CustomDataset
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import torch
import numpy as np


def train(model, num_epochs, train_loader, optimizer, start_epoch):
    loss_arr = []
    model.zero_grad()
    scalar = GradScaler()
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        batch_iterator = tqdm(
            train_loader, desc=f"Training,epoch={epoch+1}", leave=True
        )

        total_loss = []
        for count, (x, y, filename) in enumerate(batch_iterator):

            # x_pil = T.ToPILImage()((x[0] + 1) / 2)
            # y_pil = T.ToPILImage()(y[0])
            # x_pil.save("x.jpg")
            # y_pil.save("y.jpg")
            # print(filename[0])

            # raise ValueError("Stop here")

            model.train()
            x = x.to(device)
            y = y.to(device)

            with autocast():
                out = model(x)
                # print(out.min(), out.max())
                loss = torch.nn.MSELoss()(out, y)
                # loss = torch.nn.BCEWithLogitsLoss()(out, y)
            scalar.scale(loss).backward()
            scalar.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scalar.step(optimizer)
            scalar.update()
            optimizer.zero_grad()
            total_loss += [loss.item()]

            batch_iterator.set_postfix(loss=f"{np.mean(total_loss):.4f}")

        # avg_training_loss = total_loss / len(train_loader)
        # print(f"Epoch [{epoch+1}] training Loss: {avg_training_loss:.4f}")

        val_loss = eval(model, val_loader, epoch)
        # scheduler.step(val_loss)
        # loss_arr.append((epoch + 1, avg_training_loss, val_loss))

        path = f"model_bce_0-1_{epoch + 1}.pt"
        if epoch % 1 == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                },
                path,
            )
            print(f"Model saved to {path}")

        # print(loss_arr)

    # loss_array = np.array(loss_arr)
    # np.savetxt(
    #     "loss.txt",
    #     loss_array,
    #     header="epoch, avg_training_loss, avg_val_loss",
    #     delimiter=", ",
    #     fmt="%s",
    # )


def eval(model, val_loader, epoch, device="cuda"):
    model.eval()
    num_correct = 0
    num_pixels = 0
    total_loss = 0
    with torch.no_grad():
        val_iterator = tqdm(val_loader, desc=f"Evaluating Epoch {epoch + 1}")
        for x, y, filename in val_iterator:
            x = x.to(device)
            y = y.to(device)

            # 啟用 AMP 模式
            with autocast():
                out = model(x)
                # loss = torch.nn.MSELoss()(out, y)
                # loss = torch.nn.BCEWithLogitsLoss()(out, y)
                probability = torch.sigmoid(out)
                predictions = (probability > 0.5).float()

            # 計算準確度
            num_correct += (predictions == y).sum().item()
            num_pixels += torch.numel(predictions)
            # total_loss += loss.item()

    accuracy = num_correct / num_pixels
    avg_loss = total_loss / len(val_loader)

    print(
        f"Epoch [{epoch + 1}] Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
    )

    return avg_loss


if __name__ == "__main__":
    # Check the device we are using is GPU or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)

    model = UNet().cuda()
    # Choosing Adam as our optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # model_path = 'model_1104_30.pt'
    # checkpoint = torch.load(model_path)

    # model.load_state_dict(checkpoint['model_state_dict'])

    # # 加载优化器状态
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = 1e-4
    # Constants for UNet model training process
    BATCH_SIZE = 32
    NUM_EPOCHS = 40
    START_EPOCH = 0
    IMG_WIDTH = 512
    IMG_HEIGHT = 512
    # Load data
    transform = T.Compose(
        [
            T.ToTensor(),
            # T.RandomHorizontalFlip(),
            # T.RandomVerticalFlip(),
            T.Resize((IMG_HEIGHT, IMG_WIDTH), antialias=True),
        ]
    )
    train_dataset = CustomDataset(
        "/data1/raytsai/data", "/data1/raytsai/data_mask", transform=transform
    )
    val_dataset = CustomDataset(
        "/data1/raytsai/test", "/data1/raytsai/test_mask", transform=transform
    )

    # 創建 DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    # 定義 scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)

    train(model, NUM_EPOCHS, train_loader, optimizer, start_epoch=START_EPOCH)
