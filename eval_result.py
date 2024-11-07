import torch
import numpy as np
import os
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from unet_utils import CustomDataset, UNet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
import random
import shutil

test_num = 1
def load_model(device):
    path = 'model_1107_3.pt'
    model = UNet()
    model = model.to(device)  
    model.load_state_dict(torch.load(path, map_location=device)['model_state_dict'])
    return model

def save_output_as_image(tensor, output_dir, filename):
    # 将预测结果从 [-1, 1] 转换回 [0, 1]
    output_image = tensor.squeeze().cpu().numpy()
    # output_image = (output_image + 1) / 2  # 先转回[0,1]
    print(output_image)
    binary_mask = (output_image > 0.5).astype(np.uint8) * 255  # 二值化并转换为0-255范围
    image = Image.fromarray(binary_mask)
    image.save(os.path.join(output_dir, filename))



def evaluate_model(model, test_loader,  output_dir='output_images', device='cuda'):
    model.eval()
    total_loss = 0
    num_correct = 0
    num_pixels = 0
    
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用 tqdm 顯示進度條
    with torch.no_grad():
        for idx, (x, y, filename) in enumerate(tqdm(test_loader, desc="Evaluating", unit="batch")):
            x = x.to(device)
            y = y.to(device)
            
            # 推論
            output = model(x)
            output = torch.sigmoid(output)
            print(output)
            predictions = (output > 0.5).float()
            
            
            # 計算準確度 (Accuracy)
            num_correct += (predictions == y).sum().item()
            num_pixels += torch.numel(predictions)
            
            # 儲存每張圖片的推論結果
            for i in range(predictions.size(0)):
                save_output_as_image(predictions[i], output_dir, f"out_{filename[i]}")
    
    accuracy = num_correct / num_pixels
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy

def sort_nicely(file_list):
    """按自然顺序对文件名进行排序"""
    import re
    return sorted(file_list, key=lambda x: [int(i) if i.isdigit() else i for i in re.split(r'(\d+)', x)])


def combine_images(input_folder, label_folder, model_output_folder, output_folder):
    # 获取文件夹中的所有图像文件
    images_info = [
       (723, 668), (667, 654),(583, 678),
       (710, 797), (600, 722), (840, 703),
       (710, 683), (773, 645), (738, 820)
    ]
    input_images = sort_nicely([f for f in os.listdir(input_folder) if f.endswith(('.jpg'))])
    label_images = sort_nicely([f for f in os.listdir(label_folder) if f.endswith(('.jpg'))])
    model_output_images = sort_nicely([f for f in os.listdir(model_output_folder) if f.endswith(('.jpg'))])

    # 确保文件夹中的图像数量相同
    num_images = min(len(input_images), len(label_images), len(model_output_images))

    if num_images == 0:
        print("没有找到任何图像，请检查文件夹是否包含图像。")
        return

    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)
    target_width, target_height = images_info[test_num-1]
    for i in range(num_images):
        # 构造每张图像的完整路径
        input_image_path = os.path.join(input_folder, input_images[i])
        label_image_path = os.path.join(label_folder, label_images[i])
        model_output_image_path = os.path.join(model_output_folder, model_output_images[i])

        # 读取图片
        input_image = cv2.imread(input_image_path)
        label_image = cv2.imread(label_image_path)
        model_output_image = cv2.imread(model_output_image_path)

        # 检查每张图片是否成功读取
        if input_image is None or label_image is None or model_output_image is None:
            print(f"无法读取以下图像之一: {input_image_path}, {label_image_path}, {model_output_image_path}")
            continue

        # 确保图片格式一致，转成 RGB
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)
        model_output_image = cv2.cvtColor(model_output_image, cv2.COLOR_BGR2RGB)

        # 调整图像大小为 560x512
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
        height, width, _ = input_image_padded.shape  # 获取图像高度和宽度
        offset = 10  # 设置偏移量

        # 添加文字标记，在每个图像的左下角
        cv2.putText(input_image_padded, input_images[i], (10, height - offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(label_image_padded, label_images[i], (10, height - offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(model_output_image_padded, model_output_images[i], (10, height - offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


        # 合并三张图片
        combined_image = np.hstack((input_image_padded, label_image_padded, model_output_image_padded))

        # 构造输出文件名，使用正确的索引
        output_filename = f"combined_{input_images[i]}"
        output_path = os.path.join(output_folder, output_filename)

        # 保存合并后的图片
        cv2.imwrite(output_path, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))
        # print(f"Combined image saved to {output_path}")

def split_image():
    data_folder = 'data'
    data_mask_folder = 'data_mask'
    test_folder = 'test'
    test_mask_folder = 'test_mask'
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(test_mask_folder, exist_ok=True)

    # 获取所有文件夹中所有符合条件的图像
    image_files = [f for f in os.listdir(data_folder) if f.endswith('.jpg')]

    # 随机选择500张图片
    selected_images = random.sample(image_files, min(500, len(image_files)))

    for filename in selected_images:
        # 复制图像到测试文件夹
        src = os.path.join(data_folder, filename)
        dst = os.path.join(test_folder, filename)
        shutil.copy(src, dst)

        # 复制对应的掩码图像到测试掩码文件夹
        mask_filename = filename.replace('.jpg', '_mask.jpg')
        mask_src = os.path.join(data_mask_folder, mask_filename)
        mask_dst = os.path.join(test_mask_folder, mask_filename)
        shutil.copy(mask_src, mask_dst)

if __name__ == '__main__':
    # 設定裝置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 載入訓練好的模型
    model = load_model(device=device)
    os.system('rmdir /s /q output_combine output_images test test_mask')
    
    # 调用 split_image 函数来选择500张图片
    split_image()

    # 加载测试数据集    
    test_data = CustomDataset('test', 'test_mask', transform=T.Compose([T.ToTensor(), T.Resize((512, 512))]))
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False)

    # 进行模型评估并保存输出结果
    evaluate_model(model, test_loader, output_dir='output_images', device=device)

    # 合并图片
    combine_images('test', 'test_mask', 'output_images', 'output_combine')