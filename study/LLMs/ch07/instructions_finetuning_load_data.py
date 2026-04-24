from importlib.metadata import version

pkgs = [
    "matplotlib",  # 绘图库
    "tiktoken",  # 分词器
    "torch",  # 深度学习库
    "tqdm",  # 进度条
    # "tensorflow",  # 用于加载OpenAI的预训练权重
]
for p in pkgs:
    print(f"{p} version: {version(p)}")
# 读取并输出版本号

import json
import os
import urllib
import urllib.request  # 修改这里

def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


# 在网上下载并打开数据库

file_path = "instruction-data.json"
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)

data = download_and_load_file(file_path, url)
print("Number of entries:", len(data))
# 看一下数据一共有多少条
