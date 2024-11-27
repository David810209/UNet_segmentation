from PIL import Image, ImageDraw, ImageFont

import os


# 設定字體和尺度條資訊
font_path = "arial.ttf"  # 依系統調整字體檔案路徑
font_size = 20
font = ImageFont.truetype(font_path, font_size)
scale_bar_length = 100  # 尺度條長度（像素）
scale_text = "130 μm"


image = Image.open("test_mask/1_199_mask.jpg")
draw = ImageDraw.Draw(image)

# 計算文字位置
text_bbox = draw.textbbox((0, 0), scale_text, font=font)
text_width = text_bbox[2] - text_bbox[0]
text_height = text_bbox[3] - text_bbox[1]

# 設定尺度條和文字的位置
bar_x_start = image.width - scale_bar_length 
bar_y_start = image.height - 40
bar_x_end = image.width - 20
bar_y_end = bar_y_start
text_x_position = bar_x_start
text_y_position = bar_y_start - text_height - 5

# 畫尺度條
draw.line([(bar_x_start, bar_y_start), (bar_x_end, bar_y_end)], fill="black", width=5)

# 加入文字標示
draw.text((text_x_position, text_y_position), scale_text, fill="black", font=font)

# 儲存圖片
image.save("tm.jpg")

print("所有圖片已處理完成並儲存在 'output_with_scale' 資料夾中。")
