import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

input_dir="output_bce_-1-1_b8_40_single"
concat_dir="output_bce_-1-1_b8_40_combine"
test_select = 2
output_dir= f"output_{test_select}"
csv_path= f"output_analysis-{test_select}.csv"

def fill_and_color_holes(binary_image, num_labels, labels, stats, background_color=(255, 255, 255)):
    """
    根據孔洞大小進行著色。
    """
    output_image = np.ones((labels.shape[0], labels.shape[1], 3), dtype=np.uint8) * np.array(background_color, dtype=np.uint8)

    # 定義不同大小範圍及其對應顏色 (B, G, R 格式)
    size_to_color = {
        "small": (255, 0, 0),          # 小孔洞 (紅色)
        "medium": (255, 255, 0),       # 中等孔洞 (黃色)
        "large": (0, 255, 0),          # 大孔洞 (綠色)
        "extra_large": (0, 0, 255)     # 超大孔洞 (藍色)
    }
    thresholds = [20, 50, 200]  # 面積閾值，小、中、大的上限

    for label in range(num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if label == 0:  # 背景處理
            color = background_color
        elif area <= thresholds[0]:
            color = size_to_color["small"]
        elif area <= thresholds[1]:
            color = size_to_color["medium"]
        elif area <= thresholds[2]:
            color = size_to_color["large"]
        else:
            color = size_to_color["extra_large"]
        output_image[labels == label] = color

    return output_image


def process_and_merge_images(center_ratio=0.65, offset_x=-50, offset_y=50):
    os.makedirs(output_dir, exist_ok=True)

    thresholds = [20, 50, 200]  # 面積閾值
    results = []  # 用於保存 CSV 統計結果

    input_files = [f for f in os.listdir(input_dir) if f.startswith(f"concat_{test_select}") and f.endswith(".jpg")]

    for input_file in tqdm(input_files, desc="Processing images"):
        concat_file = f"{input_file}"
        input_path = os.path.join(input_dir, input_file)
        concat_path = os.path.join(concat_dir, concat_file)

        if not os.path.exists(concat_path):
            print(f"未找到對應的 concat 文件: {concat_file}")
            continue

        # 讀取 input_dir 中的灰度圖片
        gray_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if gray_image is None:
            print(f"無法讀取灰度圖像: {input_file}")
            continue
        # 設定圓形區域參數 (偏移圓心)
        h, w = gray_image.shape
        center_x, center_y = w // 2 + offset_x, h // 2 + offset_y  # 偏移圓心
        radius = int(min(h, w) * center_ratio / 2)

        # 創建遮罩
        mask = np.zeros_like(gray_image, dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)  # 填充圓形區域

        # 將遮罩應用到原始圖像，獲取圓形區域
        circle_region = cv2.bitwise_and(gray_image, gray_image, mask=mask)

        # 確保是二值圖
        _, binary_image = cv2.threshold(circle_region, 127, 255, cv2.THRESH_BINARY_INV)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
        # 進行孔洞著色
        color_image = fill_and_color_holes(binary_image,  num_labels, labels, stats)

        # 讀取 concat_dir 中的三連圖
        concat_image = cv2.imread(concat_path)
        if concat_image is None:
            print(f"無法讀取三連圖像: {concat_file}")
            continue

        # 調整高度匹配
        if color_image.shape[0] != concat_image.shape[0]:
            color_image = cv2.resize(color_image, (color_image.shape[1], concat_image.shape[0]))

        # 水平合併兩張圖片
        merged_image = np.hstack((concat_image, color_image))
        output_path = os.path.join(output_dir, f"{os.path.splitext(input_file)[0]}_merged.jpg")
        cv2.imwrite(output_path, merged_image)

        # 統計數據
        max_hole_size=30000
        for label in range(1, num_labels):  # 跳過背景 (label 0)
            mask = (labels == label)
            hole_size = stats[label, cv2.CC_STAT_AREA]
            average_gray = np.mean(gray_image[mask])

            if hole_size <= max_hole_size:
                results.append({"hole_size": hole_size, "average_gray_value": average_gray})
        

    # 分組統計
    group_size=50
    df = pd.DataFrame(results)
    df['size_group'] = (df['hole_size'] // group_size) * group_size  # 分組 (e.g., 0-49, 50-99, etc.)
    grouped = df.groupby('size_group').mean().reset_index()

    # 保存結果到 CSV
    grouped.to_csv(csv_path, index=False)
    print(f"分組分析結果已保存到: {csv_path}")

def plot_grouped_hole_analysis(csv_path):
    """
    根據分組 CSV 繪製統計圖。
    """
    # 讀取數據
    df = pd.read_csv(csv_path)
    # 繪製長條圖
    plt.figure(figsize=(10, 6))
    plt.bar(df['size_group'], df['average_gray_value'], width=40)
    plt.xlabel("Hole Size Group")
    plt.ylabel("Average Grayscale Value")
    plt.title("Grouped Average Grayscale Value per Hole Size")
    plt.tight_layout()
    plt.show()


process_and_merge_images(center_ratio=0.65, offset_x=-50, offset_y=50)
plot_grouped_hole_analysis("output_analysis-1.csv")
