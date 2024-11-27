import numpy as np
import matplotlib.pyplot as plt

# 讀取數據
data = np.loadtxt('loss.txt', delimiter=',', skiprows=1)

# 提取每列數據
epochs = data[:, 0]
# total_loss = data[:, 1]
avg_training_loss = data[:, 1]
val_loss = data[:, 2]

# 繪製折線圖
plt.figure(figsize=(10, 6))
# plt.plot(epochs, total_loss, label='Total Loss')
plt.plot(epochs, avg_training_loss, label='Average Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')

# 添加標籤和標題
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)

# 顯示圖表
plt.show()
