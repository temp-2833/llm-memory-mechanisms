# 按照category进行分组统计
import json
import pandas as pd

# 打开之前评估脚本生成的评估指标文件
with open("evaluation_metrics.json", "r") as f:
    data = json.load(f)

# 将按技术分类的嵌套结构转换为平铺的列表，便于分析
all_items = []
for key in data:
    all_items.extend(data[key])

df = pd.DataFrame(all_items)

df["category"] = pd.to_numeric(df["category"])

# 按category计算分数平均值
# result = df.groupby("category").agg({"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"}).round(4)
result = df.groupby("category").agg({"bleu_score": "mean", "f1_score": "mean"}).round(4)

# 统计每个类别下有多少个问题
result["count"] = df.groupby("category").size()

print("每个Category的平均分数:")
print(result)

# 计算总体平均值
# overall_means = df.agg({"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"}).round(4)
overall_means = df.agg({"bleu_score": "mean", "f1_score": "mean"}).round(4)

print("\n总体平均分数:")
print(overall_means)
