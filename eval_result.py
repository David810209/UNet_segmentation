import torch
import numpy as np
import os
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from unet_utils import CustomDataset, UNet
from torch.utils.data import DataLoader
import cv2
import shutil

# 設定模型檔案路徑
model_name = 'bce_-1-1_b8_40'
model_path = f"model_{model_name}.pt"

images_info = [
        (723, 668),
        (667, 654),
        (583, 678),
        (710, 797),
        (600, 722),
        (840, 703),
        (710, 683),
        (773, 645),
        (738, 820),
    ]

def compute_iou(pred, target):
    """
    計算 IoU (Intersection over Union)
    Args:
        pred (torch.Tensor): 預測二值張量
        target (torch.Tensor): 真值二值張量
    Returns:
        float: IoU 值
    """
    # 確保 pred 和 target 是布爾類型
    pred = pred.bool()
    target = target.bool()

    # 計算交集和聯集
    intersection = (pred & target).float().sum()  # 交集
    union = (pred | target).float().sum()        # 聯集

    if union == 0:
        return 0.0  # 避免除以零
    return intersection / union




def load_model(device):
    # 載入訓練好的 UNet 模型
    model = UNet().to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device)["model_state_dict"]
    )
    return model

def evaluate_model(model, test_loader, output_dir,output_dir2, device="cuda"):
    # 評估模型並將結果保存為圖像
    model.eval()

    losses = []
    iou_scores = []
    with torch.no_grad():
        for idx, (x, y, filename) in enumerate(
            tqdm(test_loader, desc="Evaluating", unit="batch")
        ):
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            output = (output > 0.5).float()

            for i in range(x.size(0)):
                iou = compute_iou(output[i], y[i])  # 計算每張影像的 IoU
                iou_scores.append(iou.cpu().numpy())

                file_start_number = int(filename[i].split("_")[0])
                width, height = images_info[file_start_number - 1]  # 确保宽高正确

                # 恢复 x_img 到 [0, 1] 范围并扩展到 3 通道
                x_img = (x[i] + 1) / 2  # 恢复到 [0, 1]

                # 确保 y_img 和 output_img 与 x_img 一致，并扩展到 3 通道
                y_img = y[i].repeat(3, 1, 1)
                # y_img = y[i]
                output_img = output[i].repeat(3, 1, 1)
                # output_img = output[i]
                # 调整大小，确保张量维度一致
                x_img = T.Resize((height, width))(T.ToPILImage()(x_img.cpu()))
                y_img = T.Resize((height, width))(T.ToPILImage()(y_img.cpu()))
                output_img = T.Resize((height, width))(T.ToPILImage()(output_img.cpu()))

                x_img = T.ToTensor()(x_img)
                y_img = T.ToTensor()(y_img)
                output_img = T.ToTensor()(output_img)

                # 拼接图像
                img = output_img
                img_pil = T.ToPILImage()(img.cpu()).resize((width, height))  # Resize 拼接后的图像
                img_pil.save(os.path.join(output_dir, f"concat_{filename[i]}"))

                img2 = torch.cat((x_img, y_img, output_img), dim=2)  # 沿宽度拼接
                img_pil2 = T.ToPILImage()(img2.cpu()).resize((width * 3, height))  # Resize 拼接后的图像
                img_pil2.save(os.path.join(output_dir2, f"concat_{filename[i]}"))


            # 計算損失
            loss = torch.nn.BCEWithLogitsLoss()(output, y)
            losses.append(loss.item())

    print(f"Average Loss: {np.mean(losses):.4f}")
    print(f"Average IoU: {np.mean(iou_scores):.4f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(device=device)
    output_dir = f"output_tmp_single"
    output_dir2  = f"output_tmp_combine"
    # 清除之前的結果
    for folder in [output_dir, output_dir2]:
        if os.path.exists(folder):
            shutil.rmtree(folder)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir2, exist_ok=True)

    # 加載測試數據
    test_data = CustomDataset(
        "test_data",
        "test_data_mask",
        transform=T.Compose([T.ToTensor(), T.Resize((512, 512))]),
    )
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # 模型評估
    evaluate_model(model, test_loader,output_dir,output_dir2)

# 加載測試數據
    # test_data2 = CustomDataset(
    #     "test_data",
    #     "test_data_mask",
    #     transform=T.Compose([T.ToTensor(), T.Resize((512, 512))]),
    # )
    # test_loader2 = DataLoader(test_data2, batch_size=1, shuffle=False)

    # # 模型評估
    # evaluate_model(model, test_loader2,output_dir,output_dir2)
