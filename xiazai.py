import os

# 本地和远程路径
source_dir = "/mnt/gaojuanru/EchoSight/Questions/InfoSeek/huggingface/all/"
target_dir = "10.3.208.26:/home/gaojuanru/data_link/ViDoRAG/data/data/infoseek/img/"

# 需要复制的文件列表
files_to_download = [
    "oven_04952590.jpg"
]

# 遍历文件并使用 scp 传输
for file in files_to_download:
    src_path = os.path.join(source_dir, file)
    scp_command = f"scp {src_path} {target_dir}"
    print(f"执行命令: {scp_command}")
    os.system(scp_command)  # 运行 scp 命令

print("文件传输完成！")