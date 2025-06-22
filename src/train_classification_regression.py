# src/train_classification_regression.py

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
from src.load_data import load_datasets
from src.feature_extraction import build_feature_matrix

def main():
    # 1) 读取并提取特征
    kin, strain = load_datasets()
    X = build_feature_matrix(kin)
    y_reg = strain['90PercStrain']
    
    # 2) 构造分类标签：当 MPS90 > 阈值 0.2 视为高风险
    threshold = 0.2
    y_clf = (y_reg > threshold).astype(int)
    
    # 3) 划分训练/测试集，并按分类标签做分层
    X_tr, X_te, y_clf_tr, y_clf_te, y_reg_tr, y_reg_te = train_test_split(
        X, y_clf, y_reg, test_size=0.2, random_state=42, stratify=y_clf
    )
    
    # 4) 风险分类器：XGBClassifier
    clf = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    clf.fit(X_tr, y_clf_tr)
    preds_clf = clf.predict(X_te)
    preds_proba = clf.predict_proba(X_te)[:, 1]
    
    acc = accuracy_score(y_clf_te, preds_clf)
    auc = roc_auc_score(y_clf_te, preds_proba)
    print(f"Classification – Accuracy: {acc:.4f}, ROC AUC: {auc:.4f}")
    
    # 5) 在预测为高风险的样本上做回归
    high_idx_tr = y_clf_tr == 1
    high_idx_te = preds_clf == 1
    
    if high_idx_tr.sum() > 0 and high_idx_te.sum() > 0:
        reg = XGBRegressor(random_state=42)
        reg.fit(X_tr[high_idx_tr], y_reg_tr[high_idx_tr])
        preds_reg = reg.predict(X_te[high_idx_te])
        
        from math import sqrt
        mse  = mean_squared_error(y_reg_te[high_idx_te], preds_reg)
        rmse = sqrt(mse)
        r2   = r2_score(y_reg_te[high_idx_te], preds_reg)
        print(f"High-risk regression – RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        # 保存高风险回归模型
        os.makedirs("models", exist_ok=True)
        joblib.dump(reg, "models/highrisk_regressor.pkl")
        print("Saved high-risk regressor to models/highrisk_regressor.pkl")
    else:
        print("No high-risk samples in train or test split; adjust threshold or data.")
    
    # 保存分类模型
    joblib.dump(clf, "models/risk_classifier.pkl")
    print("Saved classifier to models/risk_classifier.pkl")

if __name__ == "__main__":
    main()
