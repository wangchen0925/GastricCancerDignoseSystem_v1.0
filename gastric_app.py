# ==============================================================================
# 胃癌基因智能诊断系统 - 全Python最终版（彻底修复特征不匹配报错 + 多格式支持）
# 文件名：gastric_app_pure_python_fixed_v2.py
# 修复内容：
# 1. 彻底解决特征名/顺序不匹配：1:1复刻R代码特征工程逻辑，保证训练/预测特征完全一致
# 2. 支持CSV/TSV/TXT多格式上传，手动选择分隔符
# 3. 修复dropna轴错误，正确删除含缺失值的样本
# 4. 严格按训练顺序匹配特征，禁止用set打乱顺序
# 5. 完善异常校验与日志，便于排查问题
# ==============================================================================
import os
import json
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.io as pio
from flask import Flask, request, jsonify, render_template_string
import traceback
from io import StringIO

# ====================== Flask 初始化 ======================
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB文件限制

# ====================== 工作目录校验 ======================
WORK_DIR = r"F:/gastric_diagnose"
if not os.path.exists(WORK_DIR):
    print(f"❌ 工作目录不存在：{WORK_DIR}")
    raise FileNotFoundError(f"目录不存在：{WORK_DIR}")
os.chdir(WORK_DIR)
print(f"✅ 工作目录设置成功：{WORK_DIR}")

# ====================== 模型加载（严格保证顺序与训练时一致） ======================
# 【重要】需从R导出3个文件到model目录：
# 1. rf_genes.csv：模型特征基因列表（一行一个，顺序与训练时完全一致）
# 2. diff_data.csv：差异表达数据（第一列为基因名索引，顺序与训练时完全一致）
# 3. ann_model.joblib：sklearn训练的ANN模型（必须用本代码相同逻辑生成的X_train训练）
try:
    rf_genes = pd.read_csv(os.path.join(WORK_DIR, "model/rf_genes.csv"), header=None)[0].tolist()
    print(f"✅ 加载rf_genes，共{len(rf_genes)}个特征基因（顺序与训练时一致）")

    diff_data = pd.read_csv(os.path.join(WORK_DIR, "model/diff_data.csv"), index_col=0)
    print(f"✅ 加载diff_data，共{len(diff_data)}个差异基因（顺序与训练时一致）")

    ann_model = joblib.load(os.path.join(WORK_DIR, "model/ann_model.joblib"))
    print("✅ 加载ANN诊断模型成功")
    if hasattr(ann_model, 'n_features_in_'):
        print(f"ℹ️ 模型训练时的特征数：{ann_model.n_features_in_}")

except Exception as e:
    print(f"❌ 模型加载失败：{str(e)}")
    traceback.print_exc()
    raise

# 诊断历史记录
diagnosis_records = []


# ====================== 核心诊断函数（1:1复刻R代码逻辑，彻底修复特征不匹配） ======================
def gastric_diagnose(patient_data):
    """
    输入：patient_data -> pd.DataFrame，行=样本，列=基因，第一列为样本ID（索引）
          （上传路由已自动将「行=基因，列=样本」的输入转置为此格式）
    输出：list[dict]，每个样本的诊断结果（sample_id, result, probability）
    """
    # ==============================================
    # 🔧 修复1：严格按训练顺序匹配特征，禁止用set打乱顺序
    # ==============================================
    # 1. 校验输入包含所有模型必需基因，缺一不可
    missing_genes = [gene for gene in rf_genes if gene not in patient_data.columns]
    if len(missing_genes) > 0:
        raise ValueError(
            f"❌ 输入数据缺失模型必需的特征基因！\n"
            f"缺失基因（前10个）：{', '.join(missing_genes[:10])}\n"
            f"共缺失{len(missing_genes)}个基因，请检查：\n"
            f"1. 基因名大小写/拼写是否正确\n"
            f"2. 输入文件格式是否为「行=基因，列=样本」\n"
            f"3. 是否包含所有模型训练时用的rf_genes"
        )

    # 2. 严格按rf_genes顺序提取特征（保证列顺序与训练时完全一致）
    patient_data_ordered = patient_data[rf_genes]
    print(f"✅ 按模型特征顺序提取{len(rf_genes)}个基因，无缺失")

    # 3. 转置为「行=基因，列=样本」（对应R的t()操作，行顺序与rf_genes一致）
    patient_exp = patient_data_ordered.T
    print(f"✅ 转置完成，基因数：{len(patient_exp)}，样本数：{len(patient_exp.columns)}")

    # ==============================================
    # 🔧 修复2：严格按diff_data训练顺序匹配基因
    # ==============================================
    diff_genes_in_order = diff_data.index.tolist()
    missing_diff_genes = [gene for gene in diff_genes_in_order if gene not in patient_exp.index]
    if len(missing_diff_genes) > 0:
        raise ValueError(f"❌ 差异基因列表中存在输入缺失的基因：{', '.join(missing_diff_genes)}")

    # 按diff_data顺序提取表达量（行顺序与训练时完全相同）
    patient_exp_diff = patient_exp.loc[diff_genes_in_order]
    print(f"✅ 按差异基因顺序提取{len(diff_genes_in_order)}个基因，顺序与训练时一致")

    # ==============================================
    # 🔧 修复3：1:1复刻R代码的二值化得分逻辑，还原训练特征结构
    # ==============================================
    # 分离上调/下调基因（顺序与diff_data完全一致）
    up_mask = diff_data['logFC'] > 0
    down_mask = diff_data['logFC'] < 0

    up_exp = patient_exp_diff.loc[up_mask]
    down_exp = patient_exp_diff.loc[down_mask]
    print(f"✅ 分离上调基因{len(up_exp)}个，下调基因{len(down_exp)}个")

    # 计算上调基因二值化得分：x > 中位数 → 1，否则0（复刻R代码apply逻辑）
    up_median = up_exp.median(axis=1)
    score_up = (up_exp.T > up_median).astype(int)  # 转置后行=样本，列=上调基因

    # 计算下调基因二值化得分：x > 中位数 → 0，否则1（复刻R代码apply逻辑）
    down_median = down_exp.median(axis=1)
    score_down = (down_exp.T <= down_median).astype(int)  # 转置后行=样本，列=下调基因

    # 合并特征矩阵（顺序与训练时完全一致：上调基因在前，下调基因在后）
    X_predict = pd.concat([score_up, score_down], axis=1)
    print(f"✅ 构建预测特征矩阵，样本数：{len(X_predict)}，特征数：{len(X_predict.columns)}")

    # ==============================================
    # 🔧 修复4：缺失值处理：删除含缺失值的样本（axis=0删除行/样本）
    # ==============================================
    if X_predict.isnull().any().any():
        original_count = len(X_predict)
        X_predict = X_predict.dropna(axis=0)
        removed_count = original_count - len(X_predict)
        print(f"⚠️ 检测到缺失值，删除{removed_count}个含缺失值的样本")

    if len(X_predict) == 0:
        raise ValueError("❌ 处理后无有效样本，请检查输入数据")

    # ==============================================
    # 🔧 修复5：模型预测（特征顺序与训练时100%一致）
    # ==============================================
    pred_proba = ann_model.predict_proba(X_predict)[:, 1]  # 取癌症（阳性）概率
    pred_label = ann_model.predict(X_predict)

    # ==============================================
    # 生成诊断结果
    # ==============================================
    results = []
    for idx, sample_id in enumerate(X_predict.index):
        result_item = {
            "sample_id": str(sample_id),
            "result": "胃癌阳性" if pred_label[idx] == 1 else "胃癌阴性",
            "probability": round(float(pred_proba[idx]), 3)
        }
        results.append(result_item)
        diagnosis_records.append(result_item)

    print(f"✅ 诊断完成，生成{len(results)}个样本结果")
    return results


# ====================== 网页界面（支持多格式上传） ======================
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>胃癌基因智能诊断系统</title>
    <style>
        * { box-sizing: border-box; }
        body { font-family: "Microsoft YaHei", sans-serif; margin: 30px; max-width: 1200px; 
               margin-left: auto; margin-right: auto; background-color: #f5f7fa; }
        h1, h2, h3 { color: #2c3e50; }
        h1 { border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        .upload-box { background: white; padding: 25px; border-radius: 10px; margin: 30px 0; 
                       box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; background: white; 
                border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        th, td { padding: 15px; text-align: left; border-bottom: 1px solid #eee; }
        th { background-color: #3498db; color: white; }
        button { background: #3498db; color: white; border: none; padding: 12px 25px; 
                 border-radius: 6px; cursor: pointer; font-size: 16px; }
        button:hover { background: #2980b9; }
        input, select { padding: 10px; width: 350px; border: 1px solid #ddd; border-radius: 6px; font-size: 16px; }
        .record-item { background: white; padding: 15px; margin: 10px 0; border-radius: 8px; }
        .error { color: #e74c3c; font-weight: bold; font-size: 18px; margin: 20px 0; }
        .tip { color: #27ae60; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>🏥 胃癌基因智能诊断系统</h1>
    <div class="upload-box">
        <h3>📤 上传基因表达文件</h3>
        <p class="tip">✅ 支持格式：CSV / TSV / TXT</p>
        <p class="tip">✅ 格式要求：行=基因，列=样本，第一列=基因名，第一行=样本ID</p>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="expression_file" accept=".csv,.tsv,.txt" required><br><br>
            <select name="sep">
                <option value=",">逗号分隔（CSV）</option>
                <option value="\t">制表符分隔（TSV/TXT）</option>
            </select><br><br>
            <button type="submit">开始诊断</button>
        </form>
    </div>

    {% if upload_results %}
    <h2>📋 诊断结果</h2>
    <table>
        <tr><th>样本ID</th><th>诊断结果</th><th>阳性概率</th></tr>
        {% for res in upload_results %}
        <tr>
            <td>{{ res.sample_id }}</td>
            <td>{{ res.result }}</td>
            <td>{{ res.probability }}</td>
        </tr>
        {% endfor %}
    </table>
    {% endif %}

    {% if error %}
    <div class="error">{{ error }}</div>
    {% endif %}

    <h2>📜 历史诊断记录</h2>
    {% for item in records %}
    <div class="record-item">
        {{ item.sample_id }} | {{ item.result }} | 阳性概率: {{ item.probability }}
    </div>
    {% else %}
    <p>暂无记录</p>
    {% endfor %}
</body>
</html>
'''


# ====================== 路由 ======================
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, records=diagnosis_records)


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        file = request.files.get('expression_file')
        sep = request.form.get('sep', ',')
        if not file or file.filename == '':
            return render_template_string(HTML_TEMPLATE, error="❌ 未选择文件", records=diagnosis_records)

        # 统一处理制表符转义问题
        if sep == '\\t':
            sep = '\t'

        # 读取文件（支持多格式）
        content = file.read().decode('utf-8')
        df = pd.read_csv(StringIO(content), index_col=0, sep=sep)
        print(f"✅ 读取文件成功，基因数：{len(df)}，样本数：{len(df.columns)}")

        # 转置：行=基因 → 行=样本，列=基因（符合诊断函数输入要求）
        df_process = df.T
        print(f"✅ 转置完成，样本数：{len(df_process)}，基因数：{len(df_process.columns)}")

        # 执行诊断
        results = gastric_diagnose(df_process)
        return render_template_string(HTML_TEMPLATE,
                                      upload_results=results,
                                      records=diagnosis_records)

    except Exception as e:
        error_msg = f"❌ 上传失败：{str(e)}"
        traceback.print_exc()
        return render_template_string(HTML_TEMPLATE,
                                      error=error_msg,
                                      records=diagnosis_records)


# ====================== 启动服务 ======================
if __name__ == '__main__':
    print("🚀 胃癌基因智能诊断系统启动成功！")
    print("🌐 访问地址：http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)