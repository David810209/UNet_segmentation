import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
input_dir="output_single"
output_dir= f"output_analysis"
csv_path= f"output_analysis.csv"


circle_info = {
    '1': {"ratio": 0.6, "x": -40, "y": 30},
    '2': {"ratio": 0.65, "x": 50, "y": 0},
    '3': {"ratio": 0.7, "x": 10, "y": -50},
    '4': {"ratio": 0.6, "x": 10, "y": -50},
    '5': {"ratio": 0.65, "x": 10, "y": -50},
    '6': {"ratio": 0.6, "x": 0, "y": 20},
    '7': {"ratio": 0.55, "x": 10, "y": -20},
    '8': {"ratio": 0.65, "x": 40, "y": 0},
    '9': {"ratio": 0.65, "x": 0, "y": 20}
}
def analyze_hole_with_cross_section(original_gray_image, label, stats, output_dir, expansion=15):
    """
    針對特定孔洞進行放大，並沿橫切線分析灰階值變化。
    """
    # 獲取孔洞的邊界資訊
    x, y, w, h = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP], stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]

    # 放大範圍
    expanded_x = max(0, x - expansion)
    expanded_y = max(0, y - expansion)
    expanded_w = min(original_gray_image.shape[1], x + w + expansion) - expanded_x
    expanded_h = min(original_gray_image.shape[0], y + h + expansion) - expanded_y

    # 裁剪放大區域
    cropped_region = original_gray_image[expanded_y:expanded_y + expanded_h, expanded_x:expanded_x + expanded_w]

    # 確保裁剪區域不為空
    if cropped_region.size == 0:
        print(f"裁剪區域為空，跳過處理 label {label}")
        return

    # 將裁剪的灰階圖像轉換為彩色圖像（BGR 格式）以支持彩色線條
    cropped_region_color = cv2.cvtColor(cropped_region, cv2.COLOR_GRAY2BGR)

    # 添加橫切線（綠色）
    line_y = expanded_h // 2  # 水平線位於裁剪區域的中間
    cv2.line(cropped_region_color, (0, line_y), (expanded_w - 1, line_y), (0, 255, 0), 1)

    # 提取橫切線上的灰階值
    cross_section = cropped_region[line_y, :]

    # 保存裁剪後的放大圖像
    output_image_path = os.path.join(output_dir, f"hole_{label}_zoomed.png")
    cv2.imwrite(output_image_path, cropped_region_color)
    # print(f"放大區域圖像已保存到: {output_image_path}")

    # 檢查橫切線數據是否有效
    if np.all(cross_section == 0):
        print(f"橫切線上的灰階值全為 0，檢查裁剪區域或遮罩 label {label}")
        return

    # 繪製灰階變化曲線
    plt.figure(figsize=(10, 6))
    plt.plot(cross_section, label=f'Hole {label} Cross Section')
    plt.title(f'Hole {label} Grayscale Variation Along Cross Section')
    plt.xlabel('Pixel Position Along Line')
    plt.ylabel('Grayscale Value')
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # 保存灰階變化圖
    output_plot_path = os.path.join(output_dir, f"hole_{label}_grayscale_plot.png")
    plt.savefig(output_plot_path)
    plt.close()
    print(f"灰階變化曲線已保存到: {output_plot_path}")

def analyze_hole_with_cross_section(original_gray_image, label, stats, output_dir,input_file, expansion=10):
    """
    針對特定孔洞進行放大，並沿橫切線分析灰階值變化。
    """
    # 獲取孔洞的邊界資訊
    x, y, w, h = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP], stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]

    # 放大範圍
    expanded_x = max(0, x - expansion)
    expanded_y = max(0, y - expansion)
    expanded_w = min(original_gray_image.shape[1], x + w + expansion) - expanded_x
    expanded_h = min(original_gray_image.shape[0], y + h + expansion) - expanded_y

    # 裁剪放大區域
    cropped_region = original_gray_image[expanded_y:expanded_y + expanded_h, expanded_x:expanded_x + expanded_w]

    # 確保裁剪區域不為空
    if cropped_region.size == 0:
        print(f"裁剪區域為空，跳過處理 label {label}")
        return

    # 將裁剪的灰階圖像轉換為彩色圖像（BGR 格式）以支持彩色線條
    cropped_region_color = cv2.cvtColor(cropped_region, cv2.COLOR_GRAY2BGR)

    # 添加橫切線（綠色）
    line_y = expanded_h // 2  # 水平線位於裁剪區域的中間
    cv2.line(cropped_region_color, (0, line_y), (expanded_w - 1, line_y), (0, 255, 0), 1)

    # 提取橫切線上的灰階值
    cross_section = cropped_region[line_y, :]

    # 保存裁剪後的放大圖像
    output_image_path = os.path.join(output_dir, f"{input_file}_hole_{label}_zoomed.png")
    cv2.imwrite(output_image_path, cropped_region_color)
    # print(f"放大區域圖像已保存到: {output_image_path}")

    # 檢查橫切線數據是否有效
    if np.all(cross_section == 0):
        print(f"橫切線上的灰階值全為 0，檢查裁剪區域或遮罩 label {label}")
        return

    # 繪製灰階變化曲線
    plt.figure(figsize=(10, 6))
    plt.plot(cross_section, label=f'Hole {label} Cross Section')
    plt.title(f'Hole {label} Grayscale Variation Along Cross Section')
    plt.xlabel('Pixel Position Along Line')
    plt.ylabel('Grayscale Value')
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # 保存灰階變化圖
    output_plot_path = os.path.join(output_dir, f"{input_file}_hole_{label}_grayscale_plot.png")
    plt.savefig(output_plot_path)
    plt.close()
    # print(f"灰階變化曲線已保存到: {output_plot_path}")



def fill_and_color_holes(binary_image, num_labels, labels, stats, mask, background_color=(255,255,255)):
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

    output_image[mask == 0] = (0,0,0)
    return output_image


def process_and_merge_images():
    os.makedirs(output_dir, exist_ok=True)
    limit = 0
    cur_id = 1
    results = []  # 用於保存 CSV 統計結果
    distribution_results = []
    input_files = [f for f in os.listdir(input_dir) if f.endswith(".jpg")]

    for input_file in tqdm(input_files, desc="Processing images"):
        id = input_file.split('_')[1]
        if int(id) != cur_id:
            continue
        if limit == 4:
            cur_id += 1
            limit = 0
            continue
        limit += 1
        center_ratio, offset_x, offset_y = circle_info[id]['ratio'], circle_info[id]['x'], circle_info[id]['y']
        input_path = os.path.join(input_dir, input_file)

        # 讀取 input_dir 中的灰度圖片
        gray_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if gray_image is None:
            print(f"無法讀取灰度圖像: {input_file}")
            continue
        original_file = re.sub("concat_", "", input_file)
        original_file_path = os.path.join("analysis_data", original_file)  # 假設格式為 id.jpg
        original_gray_image = cv2.imread(original_file_path, cv2.IMREAD_GRAYSCALE)
        if original_gray_image is None:
            print(f"無法讀取訓練數據灰度圖像: {original_file_path}")
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
        color_image = fill_and_color_holes(binary_image,  num_labels, labels, stats, mask)

        # 在著色圖像上標記孔洞編號
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] == 0:
                continue
            x, y, w, h = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP], stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 0, 0), 1)
            cx, cy = int(centroids[label][0]), int(centroids[label][1])
            cv2.putText(color_image, str(label), (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0,0), 2)
            analyze_hole_with_cross_section(original_gray_image, label, stats, output_dir,input_file)
        
        # 計算非孔洞區域的灰階平均值
        non_hole_mask = (binary_image == 0)
        non_hole_values = original_gray_image[non_hole_mask]
        # 統計灰階分布
        hist, bin_edges = np.histogram(non_hole_values, bins=256, range=(0, 255))
        # 保存非孔洞分布數據
        for bin_start, count in zip(bin_edges[:-1], hist):
            distribution_results.append({
                "file": input_file,
                "gray_level": int(bin_start),
                "pixel_count": count
            })
        # 統計數據
        max_hole_size=30000
        for label in range(1, num_labels):  # 跳過背景 (label 0)
          mask = (labels == label)
          hole_size = stats[label, cv2.CC_STAT_AREA]
          if not np.any(mask):
              print(f"跳過空的 mask，label: {label}")
              continue

          average_gray = np.mean(original_gray_image[mask])

          if hole_size <= max_hole_size:
              results.append({"hole_size": hole_size, "average_gray_value": average_gray})
        cv2.circle(original_gray_image, (center_x, center_y), radius, (0, 255, 0), 2)
        merged_image = np.hstack((
            cv2.cvtColor(original_gray_image, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(circle_region, cv2.COLOR_GRAY2BGR),
            color_image
        ))

        # 保存合併圖像
        merged_output_path = os.path.join(output_dir, f"merged_{input_file}")
        cv2.imwrite(merged_output_path, merged_image)


    # 分組統計
    group_size=10
    df = pd.DataFrame(results)
    df['size_group'] = (df['hole_size'] // group_size) * group_size  # 分組 (e.g., 0-49, 50-99, etc.)
    grouped = df.groupby('size_group').mean().reset_index()

    distribution_csv_path = "non_hole_distribution.csv"
    distribution_df = pd.DataFrame(distribution_results)
    distribution_df.to_csv(distribution_csv_path, index=False)

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
    combined_output_file = os.path.join(output_dir, "孔洞內灰階數分布圖.png")
    plt.savefig(combined_output_file)
    # plt.show()

def plot_non_hole_distribution(csv_path="non_hole_distribution.csv"):
    """
    視覺化非孔洞區域的灰階分布。
    """
    # 讀取分布數據
    df = pd.read_csv(csv_path)
    
    # 獲取所有唯一圖像的文件名
    files = df['file'].unique()

    # 為每個圖像繪製灰階分布圖
    # for file in files:
    #     file_df = df[df['file'] == file]
    #     plt.figure(figsize=(10, 6))
    #     plt.bar(file_df['gray_level'], file_df['pixel_count'], width=1, align='center')
    #     plt.title(f"Non-Hole Grayscale Distribution for {file}")
    #     plt.xlabel("Grayscale Level")
    #     plt.ylabel("Pixel Count")
    #     plt.tight_layout()

    #     # 保存圖表
    #     output_file = os.path.join(output_dir, f"non_hole_distribution_{file}.png")
    #     plt.savefig(output_file)
    #     plt.close()
        # print(f"灰階分布圖已保存到: {output_file}")

    # 可選：將所有圖像的灰階分布疊加在一個圖表中
    plt.figure(figsize=(12, 8))
    for file in files:
        file_df = df[df['file'] == file]
        plt.plot(file_df['gray_level'], file_df['pixel_count'], label=file)

    plt.title("Combined Non-Hole Grayscale Distribution")
    plt.xlabel("Grayscale Level")
    plt.ylabel("Pixel Count")
    plt.legend(loc='upper right', fontsize='small')
    plt.tight_layout()

    # 保存綜合視覺化
    combined_output_file = os.path.join(output_dir, "非孔洞區綜合灰階分布圖.png")
    plt.savefig(combined_output_file)
    plt.close()
    print(f"綜合灰階分布圖已保存到: {combined_output_file}")

    grouped = df.groupby('gray_level')['pixel_count'].sum().reset_index()
     # 計算平均分布
    total_images = df['file'].nunique()  # 獲取圖像總數
    grouped['average_pixel_count'] = grouped['pixel_count'] / total_images

    # 繪製平均灰階分布圖
    plt.figure(figsize=(10, 6))
    plt.bar(grouped['gray_level'], grouped['average_pixel_count'], width=1, align='center', color='blue')
    plt.title("Average Non-Hole Grayscale Distribution")
    plt.xlabel("Grayscale Level")
    plt.ylabel("Average Pixel Count")
    plt.tight_layout()

    # 保存平均分布圖
    average_output_file = os.path.join(output_dir, "非孔洞區平均灰階分布圖.png")
    plt.savefig(average_output_file)
    plt.close()

    print(f"平均非孔洞灰階分布圖已保存到: {average_output_file}")

# import shutil
# if os.path.exists("center_visualization"):
#     shutil.rmtree("center_visualization")
# visualize_center_circle(input_dir="output_single", output_dir="center_visualization")

process_and_merge_images()
plot_grouped_hole_analysis("output_analysis.csv")
plot_non_hole_distribution()