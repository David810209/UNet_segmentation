import os
import shutil
import random

# 定義資料夾路徑
data_dir = 'data'
data_mask_dir = 'data_mask'
test_dir = 'test'
test_mask_dir = 'test_mask'

# 確保 test 和 test_mask 資料夾存在，不存在則創建
os.makedirs(test_dir, exist_ok=True)
os.makedirs(test_mask_dir, exist_ok=True)

# 獲取 data 中所有圖片名稱（只保留 .jpg 結尾的文件）
images = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]

# 隨機選擇 1200 張圖片
selected_images = random.sample(images, 1200)

# 將選擇的圖片及其對應的 mask 移動到 test 和 test_mask 資料夾
for image_name in selected_images:
    # 定義圖片和 mask 的路徑
    img_path = os.path.join(data_dir, image_name)
    mask_name = image_name.replace('.jpg', '_mask.jpg')
    mask_path = os.path.join(data_mask_dir, mask_name)
    
    # 定義目標路徑
    test_img_path = os.path.join(test_dir, image_name)
    test_mask_path = os.path.join(test_mask_dir, mask_name)
    
    # 檢查對應的 mask 是否存在，並移動圖片和 mask
    if os.path.exists(mask_path):
        shutil.move(img_path, test_img_path)       # 移動圖片到 test 資料夾
        shutil.move(mask_path, test_mask_path)     # 移動 mask 到 test_mask 資料夾
        print(f"已移動 {image_name} 和 {mask_name} 到 test 和 test_mask")
    else:
        print(f"跳過 {image_name}，因為對應的 mask 文件不存在。")
