import pandas as pd
import json

# 读取 parquet 文件
df = pd.read_parquet('test.parquet')

# 转换为所需格式
converted = []
for idx, row in df.iterrows():
    converted.append({
        "id": idx,
        "Question": row["question"],
        "answer": row["final_answer"]
    })

# 保存为 JSON 文件
with open("converted_olympiad_test.json", "w", encoding="utf-8") as f:
    json.dump(converted, f, ensure_ascii=False, indent=4)

print("转换完成，结果已保存为 converted__test.json")
