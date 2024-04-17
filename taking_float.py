import re

# 打开文件
with open('data.txt', 'r', encoding='utf-8') as file:
    content = file.read()

# 正则表达式来查找所有 "Test results wb" 的结果块
results = re.findall(r"Test results wb\n\{[^\}]+\}", content)

# 遍历每个结果块
for result in results:
    # 在当前结果块中查找 "worst_accuracy" 后的浮点数
    match = re.search(r"'worst_accuracy': (\d+\.\d+)", result)
    if match:
        print(match.group(1))
