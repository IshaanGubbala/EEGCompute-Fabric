from sklearn.metrics import roc_auc_score, average_precision_score
def auroc(y, p): return float(roc_auc_score(y, p))
def auprc(y, p): return float(average_precision_score(y, p))
