import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

circle_info = {
    '1': {"ratio": 0.65, "x": 50, "y": -50},
    '2': {"ratio": 0.65, "x": 50, "y": 10},
    '3': {"ratio": 0.7, "x": 10, "y": -50},
    '4': {"ratio": 0.6, "x": 10, "y": -50},
    '5': {"ratio": 0.65, "x": 10, "y": -50},
    '6': {"ratio": 0.6, "x": 0, "y": 20},
    '7': {"ratio": 0.7, "x": 10, "y": -50},
    '8': {"ratio": 0.7, "x": 10, "y": -50},
}

    

def visualize_center_circle(input_dir="output_images", output_dir="center_circle_visualization"):
    """
    視覺化圖片的圓形中心區域，確保裁剪範圍合理。
    """
    os.makedirs(output_dir, exist_ok=True)
    input_files = [f for f in os.listdir(input_dir) if f.startswith("concat_6") and f.endswith(('.jpg', '.png', '.jpeg'))]
    done = 0
    for input_file in input_files:
        id = input_file.split("_")[1]
        center_ratio, offset_x, offset_y = circle_info[id]["ratio"], circle_info[id]["x"], circle_info[id]["y"]
        input_path = os.path.join(input_dir, input_file)
        if done:
            break
        # 讀取灰階圖像
        gray_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if gray_image is None:
            print(f"無法讀取圖像: {input_file}")
            continue

        # 計算中心圓形參數
        h, w = gray_image.shape
        center = (w // 2 + offset_x, h // 2 + offset_y )  # 圓心
        radius = int(min(h, w) * center_ratio / 2)  # 半徑為較短邊的比例

        # 創建遮罩（圓形區域）
        mask = np.zeros_like(gray_image, dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)  # 填充圓形區域

        # 將遮罩應用到原圖
        masked_image = cv2.bitwise_and(gray_image, gray_image, mask=mask)

        # 創建視覺化圖像，將圓形區域畫在原始圖像上
        visual_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        cv2.circle(visual_image, center, radius, (0, 255, 0), 2)  # 綠色圓形框

        # 保存視覺化結果
        output_path = os.path.join(output_dir, f"circle_center_{input_file}")
        cv2.imwrite(output_path, visual_image)

        # 顯示圖像（可選）
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(visual_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Circle Center Region of {input_file}")
        plt.axis("off")
        # plt.show()

    print(f"圓形中心區域視覺化結果已保存到：{output_dir}")




# 執行函數
import shutil
shutil.rmtree("center_visualization")
visualize_center_circle(input_dir="output_bce_-1-1_b8_40_single", output_dir="center_visualization")
