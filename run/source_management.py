import os
import shutil

# 定义函数，用于复制源文件夹中的所有.bmp文件到目标文件夹

def copy_bmp_files(source_folder, destination_folder):
    # 创建目标文件夹（如果不存在）
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 初始化文件序号
    file_number = 1

    # 遍历源文件夹中的所有子文件夹
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # 检查文件扩展名是否为.bmp
            if file.lower().endswith('.bmp'):
                # 构建源文件路径
                source_path = os.path.join(root, file)
                
                # 构建目标文件路径
                destination_filename = f"img_{file_number}.bmp"
                destination_path = os.path.join(destination_folder, destination_filename)
                
                # 复制文件到目标文件夹
                shutil.copy2(source_path, destination_path)
                
                # 递增文件序号
                file_number += 1

    print(f"已成功复制 {file_number - 1} 个.bmp文件到 {destination_folder} 文件夹中。")

# 指定源文件夹和目标文件夹路径
source_folder = "./sewing2d_database/source/240424/dataset"
destination_folder = "./sewing2d_database/source/240424/combined"

# 调用函数复制.bmp文件
copy_bmp_files(source_folder, destination_folder)