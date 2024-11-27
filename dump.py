import os
import shutil

def copy_files(file_list, source_dir, target_dir):
    """
    根据文件列表将文件从 source_dir 复制到 target_dir。
    
    :param file_list: 包含文件名的文件路径
    :param source_dir: 文件源目录
    :param target_dir: 文件目标目录
    """
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)

    # 读取文件列表并复制
    with open(file_list, 'r') as f:
        for line in f:
            file_name = line.strip()  # 去掉换行符或空格
            source_path = os.path.join(source_dir, file_name)
            target_path = os.path.join(target_dir, file_name)

            # 检查源文件是否存在
            if os.path.exists(source_path):
                shutil.copy2(source_path, target_path)
                print(f"Copied: {source_path} -> {target_path}")
            else:
                print(f"File not found: {source_path}")

# 主程序
def main():
    # 文件路径
    file_mappings = [
        ("train_data.txt", "all_data", "train_data"),
        ("train_data_mask.txt", "all_data_mask", "train_data_mask"),
        ("test_data.txt", "all_data", "test_data"),
        ("test_data_mask.txt", "all_data_mask", "test_data_mask"),
    ]

    # 遍历每组文件列表和目录，进行文件复制
    for file_list, source_dir, target_dir in file_mappings:
        print(f"Processing {file_list}...")
        copy_files(file_list, source_dir, target_dir)

if __name__ == "__main__":
    main()
