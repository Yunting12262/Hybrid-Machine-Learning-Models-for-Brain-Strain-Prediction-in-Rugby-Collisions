# src/train_demo_model.py

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
from src.load_data import load_datasets
from src.feature_extraction import process_sequence, extract_fft_features

def main():
    # 1) 加载数据
    kin, strain = load_datasets()
    
    # 2) 构造仅 5 个特征的 DataFrame
    df = pd.DataFrame({
        "PLA": kin["PLA"],
        "PRV": kin["PRV"],
        "PRA": kin["PRA"],
        # FFT 主频和能量，需要从时序中提取第一个轴做示例
        # 这里以 linear_acc_x 为例，若全 axis 可重复多列
        **pd.DataFrame(
            kin["linear_acc_x"]
            .apply(process_sequence)
            .apply(lambda x: extract_fft_features(x))
            .tolist()
        )
    })
    
    # 3) 分类标签
    y_reg = strain["90PercStrain"]
    y_clf = (y_reg > 0.2).astype(int)
    
    # 4) 划分
    X_tr, X_te, y_clf_tr, y_clf_te, y_reg_tr, y_reg_te = train_test_split(
        df, y_clf, y_reg, test_size=0.2, random_state=42, stratify=y_clf
    )
    
    # 5) 训练分类器
    clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    clf.fit(X_tr, y_clf_tr)
    acc = accuracy_score(y_clf_te, clf.predict(X_te))
    auc = roc_auc_score(y_clf_te, clf.predict_proba(X_te)[:,1])
    print(f"[Demo] Classification – Acc: {acc:.4f}, AUC: {auc:.4f}")
    
    # 6) 训练回归（高风险子集）
    high_idx_tr = y_clf_tr == 1
    high_idx_te = clf.predict(X_te) == 1
    if high_idx_tr.sum() and high_idx_te.sum():
        reg = XGBRegressor(random_state=42)
        reg.fit(X_tr[high_idx_tr], y_reg_tr[high_idx_tr])
        preds = reg.predict(X_te[high_idx_te])
        from math import sqrt
        rmse = sqrt(mean_squared_error(y_reg_te[high_idx_te], preds))
        r2   = r2_score(y_reg_te[high_idx_te], preds)
        print(f"[Demo] High-risk Regression – RMSE: {rmse:.4f}, R2: {r2:.4f}")
        joblib.dump(reg, "models/demo_regressor.pkl")
        print("Saved demo_regressor.pkl")
    else:
        print("No high-risk samples for demo regression.")

    # 7) 保存分类器
    joblib.dump(clf, "models/demo_classifier.pkl")
    print("Saved demo_classifier.pkl")

if __name__ == "__main__":
    main()
