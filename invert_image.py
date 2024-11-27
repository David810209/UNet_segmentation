import os
import numpy as np
from PIL import Image

# 原始標籤圖片資料夾
mask_dir = 'output_test_mask'
# 保存顛倒後標籤圖片的新資料夾
invert_mask_dir = 'invert_output_test_mask'
os.makedirs(invert_mask_dir, exist_ok=True)

# 遍歷原始標籤圖片並進行顛倒處理
for filename in os.listdir(mask_dir):
    if filename.endswith('.jpg'):
        # 讀取標籤圖片
        mask_path = os.path.join(mask_dir, filename)
        mask = np.array(Image.open(mask_path).convert('L'))

        # 黑白顛倒
        inverted_mask = 255 - mask

        # 將顛倒後的標籤保存至新資料夾
        inverted_image = Image.fromarray(inverted_mask)
        inverted_image.save(os.path.join(invert_mask_dir, filename))

print("所有標籤圖片已成功顛倒並保存到 invert_data_mask 資料夾中。")
