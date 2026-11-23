# ==============================================================================
# 批处理层 - 轻量化胃癌模型训练（Lambda架构）
# 功能：全量数据训练 → 保存核心模型/特征/指标
# 无绘图、无冗余文件、无多余日志
# ==============================================================================
setwd("F:\\gastric_diagnose")
if (!dir.exists("model")) dir.create("model")

# 极简依赖（仅核心计算包）
library(randomForest)diagnose
library(caret)
library(limma)
library(neuralnet)
library(jsonlite)
set.seed(42)

# ====================== 配置 ======================
input_config <- list(
  rf_input = "data/diffGeneExp.txt",
  diff_file = "data/diff.txt",
  test_exp = "data/GSE118916.txt"
)

# ====================== 核心工具函数 ======================
extract_group <- function(row_names) {
  sapply(strsplit(row_names, "_"), function(x) x[2])
}
preprocess_data <- function(data) {
  data <- na.omit(data)
  data[is.na(data)] <- median(data, na.rm = TRUE)
  return(data)
}

# ====================== 1. 随机森林特征筛选 ======================
data_rf <- read.table(input_config$rf_input, header=T, sep="\t", row.names=1)
data_rf <- t(data_rf)
data_rf <- preprocess_data(data_rf)
group_rf <- extract_group(rownames(data_rf))

# 轻量化RF训练
rf <- randomForest(as.factor(group_rf)~., data=data_rf, ntree=200, mtry=10, importance=TRUE)
imp <- importance(rf)[,"MeanDecreaseGini"]
rf_genes <- names(imp[imp > quantile(imp, 0.75)])
cat("筛选特征基因：", length(rf_genes), "\n")

# ====================== 2. 基因评分 ======================
diff_data <- read.table(input_config$diff_file, header=T, sep="\t", row.names=1)
exp_gs <- t(data_rf[, rf_genes])
same_gene <- intersect(rownames(exp_gs), rownames(diff_data))
exp_gs <- exp_gs[same_gene,]
diff_data <- diff_data[same_gene,]

# 计算评分
up_gs <- exp_gs[diff_data$logFC>0,]
down_gs <- exp_gs[diff_data$logFC<0,]
score_train <- rbind(
  if(nrow(up_gs)>0) t(apply(up_gs,1,function(x) ifelse(x>median(x),1,0))) else NULL,
  if(nrow(down_gs)>0) t(apply(down_gs,1,function(x) ifelse(x>median(x),0,1))) else NULL
)

# ====================== 3. ANN模型训练 ======================
ann_train <- as.data.frame(t(score_train))
ann_train$con <- ifelse(extract_group(rownames(ann_train))=="con",1,0)
ann_train$treat <- 1 - ann_train$con

# 轻量化训练
ann_model <- neuralnet(con+treat~., data=ann_train, hidden=5, 
                       act.fct="logistic", linear.output=FALSE)

# 模型评估
pred <- predict(ann_model, ann_train[,!colnames(ann_train)%in%c("con","treat")])
pred_label <- ifelse(pred[,"treat"]>=0.5,"treat","con")
true_label <- extract_group(rownames(ann_train))
cm <- confusionMatrix(factor(pred_label), factor(true_label), positive="treat")
metrics <- list(
  准确率 = round(cm$overall["Accuracy"],3),
  敏感度 = round(cm$byClass["Sensitivity"],3),
  特异度 = round(cm$byClass["Specificity"],3),
  特征基因数 = length(rf_genes)
)

# ====================== 保存轻量化模型文件 ======================
saveRDS(rf_genes, "model/rf_genes.rds")
saveRDS(ann_model, "model/ann_model.rds")
saveRDS(diff_data, "model/diff_data.rds")
write_json(metrics, "model/metrics.json", pretty=TRUE)

cat("批处理层完成！模型已保存至 /model\n")