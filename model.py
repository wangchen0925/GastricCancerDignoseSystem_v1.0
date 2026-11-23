# ==============================================================================
# 批处理层 - 轻量化胃癌模型训练（纯Python版，完全替代R）
# 功能：全量数据训练 → 保存核心模型/特征/指标
# 无绘图、无冗余文件、无多余日志
# 兼容 全Python诊断系统
# ==============================================================================
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# ====================== 初始化配置 ======================
# 工作目录（与R完全一致）
WORK_DIR = r"F:/gastric_diagnose"
os.chdir(WORK_DIR)
os.makedirs("model", exist_ok=True)

# 随机种子（与R set.seed(42) 一致）
np.random.seed(42)

# 输入文件配置（与R完全相同）
input_config = {
    "rf_input": "data/diffGeneExp.txt",
    "diff_file": "data/diff.txt",
    "test_exp": "data/GSE118916.txt"
}

# ====================== 核心工具函数（复刻R函数） ======================
def extract_group(row_names):
    """复刻R：行名按_分割，取第二个元素作为分组（con/treat）"""
    return [x.split("_")[1] for x in row_names]

def preprocess_data(data):
    """复刻R：缺失值处理 → 去全行NA，剩余NA填充中位数"""
    data = data.dropna(how="all")
    data = data.fillna(data.median())
    return data

# ====================== 1. 随机森林特征筛选（复刻R逻辑） ======================
print("🔍 开始随机森林特征筛选...")
# 读取数据 + 转置（与R t(data_rf) 一致）
data_rf = pd.read_table(input_config["rf_input"], index_col=0)
data_rf = data_rf.T
data_rf = preprocess_data(data_rf)

# 提取分组标签
group_rf = extract_group(data_rf.index)
group_rf = pd.Series(group_rf)

# 轻量化RF训练（参数与R完全一致：ntree=200, mtry=10）
rf = RandomForestClassifier(
    n_estimators=200,
    max_features=10,  # 对应R mtry=10
    random_state=42,
    oob_score=True
)
rf.fit(data_rf, group_rf)

# 计算Gini重要性，筛选>75分位数的基因（复刻R逻辑）
imp = rf.feature_importances_
genes = data_rf.columns
imp_series = pd.Series(imp, index=genes)
threshold = np.quantile(imp, 0.75)
rf_genes = imp_series[imp_series > threshold].index.tolist()

print(f"✅ 筛选特征基因：{len(rf_genes)} 个")

# ====================== 2. 基因评分计算（1:1复刻R逻辑） ======================
print("📊 计算基因表达评分...")
# 读取差异表达数据
diff_data = pd.read_table(input_config["diff_file"], index_col=0)

# 取交集基因 + 数据对齐
exp_gs = data_rf[rf_genes].T
same_gene = exp_gs.index.intersection(diff_data.index)
exp_gs = exp_gs.loc[same_gene]
diff_data = diff_data.loc[same_gene]

# 计算评分（与R逻辑完全一致）
up_gs = exp_gs[diff_data["logFC"] > 0]
down_gs = exp_gs[diff_data["logFC"] < 0]

# 上调基因：>中位数=1，否则0
score_up = up_gs.gt(up_gs.median(axis=1), axis=0).astype(int) if not up_gs.empty else pd.DataFrame()
# 下调基因：>中位数=0，否则1
score_down = 1 - down_gs.gt(down_gs.median(axis=1), axis=0).astype(int) if not down_gs.empty else pd.DataFrame()

# 合并评分矩阵
score_train = pd.concat([score_up, score_down], axis=0)

# ====================== 3. ANN模型训练（复刻R neuralnet） ======================
print("🤖 训练ANN诊断模型...")
# 构建训练集（与R格式一致）
ann_train = score_train.T

# ====================== 【修复核心】修复标签生成逻辑 ======================
# 提取分组并转为Pandas Series（解决bool无astype错误）
group_labels = pd.Series(extract_group(ann_train.index), index=ann_train.index)
ann_train["con"] = (group_labels == "con").astype(int)
ann_train["treat"] = 1 - ann_train["con"]

# 分离特征和标签
X_train = ann_train.drop(columns=["con", "treat"])
y_train = ann_train["treat"]  # 正类：treat（胃癌）

# ANN模型（复刻R neuralnet：hidden=5, logistic激活函数）
ann_model = MLPClassifier(
    hidden_layer_sizes=(5,),  # 对应R hidden=5
    activation="logistic",    # 对应R act.fct="logistic"
    solver="lbfgs",
    random_state=42,
    max_iter=1000
)
ann_model.fit(X_train, y_train)

# ====================== 模型评估（复刻R指标） ======================
print("📈 模型性能评估...")
y_pred = ann_model.predict(X_train)
y_pred_proba = ann_model.predict_proba(X_train)[:, 1]  # treat概率

# 计算混淆矩阵和指标
cm = confusion_matrix(y_train, y_pred)
tn, fp, fn, tp = cm.ravel()

accuracy = round((tp + tn) / (tp + tn + fp + fn), 3)
sensitivity = round(tp / (tp + fn), 3) if (tp + fn) != 0 else 0.0
specificity = round(tn / (tn + fp), 3) if (tn + fp) != 0 else 0.0

metrics = {
    "准确率": accuracy,
    "敏感度": sensitivity,
    "特异度": specificity,
    "特征基因数": len(rf_genes)
}

# ====================== 保存模型文件（兼容全Python诊断系统） ======================
print("💾 保存模型和特征文件...")

# 1. 保存特征基因列表
pd.Series(rf_genes).to_csv("model/rf_genes.csv", index=False, header=False)

# 2. 保存差异表达数据
diff_data.to_csv("model/diff_data.csv", index=True)

# 3. 保存ANN模型
joblib.dump(ann_model, "model/ann_model.joblib")

# 4. 保存模型指标
with open("model/metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print("🎉 批处理层完成！所有模型已保存至 /model 目录")
print(f"📊 模型性能：准确率={accuracy} | 敏感度={sensitivity} | 特异度={specificity}")