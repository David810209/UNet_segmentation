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
model_path = 'model_BCELoss_-1-1.pt'

def load_model(device):
    # 載入訓練好的 UNet 模型
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    return model

def save_output_as_image(tensor, output_dir, filename):
    # 將模型輸出轉換為灰階圖像
    output_image = tensor.squeeze().cpu().numpy()
    
    # 將輸出範圍縮放到 [0, 255] 之間
    output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())
    output_image = (output_image * 255).astype(np.uint8)
    
    # 將結果保存為灰階圖像
    image = Image.fromarray(output_image, mode='L')
    image.save(os.path.join(output_dir, filename))


# def save_output_as_image(tensor, output_dir, filename):
#     # 將模型輸出保存為圖像，並進行二值化處理
#     output_image = tensor.squeeze().cpu().numpy()
#     binary_mask = (output_image > 0.5).astype(np.uint8) * 255
#     image = Image.fromarray(binary_mask)
#     image.save(os.path.join(output_dir, filename))

def evaluate_model(model, test_loader, output_dir='output_images', device='cuda'):
    # 評估模型並將結果保存為圖像
    model.eval()
    num_correct = 0
    num_pixels = 0

    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (x, y, filename) in enumerate(tqdm(test_loader, desc="Evaluating", unit="batch")):
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            output = torch.sigmoid(output)
             # 二值化輸出
            # predictions = (output > 0.5).float()
            # 計算準確度
            # num_correct += (predictions == y).sum().item()
            # num_pixels += torch.numel(predictions)

            # # 保存每張圖片的推論結果
            # for i in range(predictions.size(0)):
            #     save_output_as_image(predictions[i], output_dir, f"out_{filename[i]}")
            
            # 打印輸出範圍
            print(f"Batch {idx}: Output Min: {output.min().item()}, Output Max: {output.max().item()}, Output Mean: {output.mean().item()}")

            # 如果你希望保存原始輸出，則可以用下面的代碼
            for i in range(output.size(0)):
                save_output_as_image(output[i], output_dir, f"raw_out_{filename[i]}")




    accuracy = num_correct / num_pixels
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy

def sort_nicely(file_list):
    """按自然顺序对文件名进行排序"""
    import re
    return sorted(file_list, key=lambda x: [int(i) if i.isdigit() else i for i in re.split(r'(\d+)', x)])

def combine_images(input_folder, label_folder, model_output_folder, output_folder):
    # 定義目標圖像大小資訊
    images_info = [
        (723, 668), (667, 654), (583, 678),
        (710, 797), (600, 722), (840, 703),
        (710, 683), (773, 645), (738, 820)
    ]

    input_images = sort_nicely([f for f in os.listdir(input_folder) if f.endswith('.jpg')])
    label_images = sort_nicely([f for f in os.listdir(label_folder) if f.endswith('.jpg')])
    model_output_images = sort_nicely([f for f in os.listdir(model_output_folder) if f.endswith('.jpg')])
    num_images = min(len(input_images), len(label_images), len(model_output_images))

    if num_images == 0:
        print("沒有找到任何圖像，請檢查文件夾是否包含圖像。")
        return
    
    test_num = input_image[0].split('_')[0]
    os.makedirs(output_folder, exist_ok=True)
    target_width, target_height = images_info[test_num-1]
    
    for i in range(num_images):
        input_image_path = os.path.join(input_folder, input_images[i])
        label_image_path = os.path.join(label_folder, label_images[i])
        model_output_image_path = os.path.join(model_output_folder, model_output_images[i])

        input_image = cv2.imread(input_image_path)
        label_image = cv2.imread(label_image_path)
        model_output_image = cv2.imread(model_output_image_path)

        if input_image is None or label_image is None or model_output_image is None:
            print(f"無法讀取以下圖像之一: {input_image_path}, {label_image_path}, {model_output_image_path}")
            continue

        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)
        model_output_image = cv2.cvtColor(model_output_image, cv2.COLOR_BGR2RGB)

        # 調整圖像大小
        input_image_resized = cv2.resize(input_image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        label_image_resized = cv2.resize(label_image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        model_output_image_resized = cv2.resize(model_output_image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

        def pad_image(image, target_size=(840, 820), color=(255, 255, 255)):
            padded_image = np.full((target_size[1], target_size[0], 3), color, dtype=np.uint8)
            height, width = image.shape[:2]
            x_offset = (target_size[0] - width) // 2
            y_offset = (target_size[1] - height) // 2
            padded_image[y_offset:y_offset + height, x_offset:x_offset + width] = image
            return padded_image

        input_image_padded = pad_image(input_image_resized)
        label_image_padded = pad_image(label_image_resized)
        model_output_image_padded = pad_image(model_output_image_resized)

        combined_image = np.hstack((input_image_padded, label_image_padded, model_output_image_padded))
        output_filename = f"combined_{input_images[i]}"
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = load_model(device=device)

    # 清除之前的結果
    for folder in ['output_combine', 'output_images']:
        if os.path.exists(folder):
            shutil.rmtree(folder)

    os.makedirs('output_images', exist_ok=True)
    os.makedirs('output_combine', exist_ok=True)

    # 加載測試數據
    test_data = CustomDataset('test', 'test_mask', transform=T.Compose([T.ToTensor(), T.Resize((512, 512))]))
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False)

    # 模型評估
    evaluate_model(model, test_loader, output_dir='output_images', device=device)

    # 合併圖片
    combine_images('test', 'test_mask', 'output_images', 'output_combine')
