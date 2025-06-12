#coding:utf-8
import json

# 读取原始数据
with open('test_origin.json', 'r', encoding='utf-8') as infile:
    original_data = json.load(infile)

# 转换数据格式
converted_data = []
for idx, item in enumerate(original_data):
    converted_item = {
        "id": idx,
        "Question": item["text"],
        "answer": item["label"]
    }
    converted_data.append(converted_item)

# 写入新的 JSON 文件
with open('test.json', 'w', encoding='utf-8') as outfile:
    json.dump(converted_data, outfile, ensure_ascii=False, indent=4)

print("转换完成，已保存为 'test.json'")
