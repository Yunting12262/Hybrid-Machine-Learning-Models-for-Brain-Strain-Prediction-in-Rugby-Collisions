# src/train_stacking.py

import os
import joblib
from src.load_data import load_datasets
from src.feature_extraction import build_feature_matrix
from sklearn.model_selection import GroupKFold, cross_val_score, train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def main():
    # 1. 加载并提取特征
    kin, strain = load_datasets()
    X = build_feature_matrix(kin)
    y = strain['90PercStrain']
    groups = kin['subject']

    # 2. 定义基模型（不在此处指定 cv）
    estimators = [
        ('xgb', XGBRegressor(random_state=42)),
        ('rf',  RandomForestRegressor(n_estimators=100, random_state=42)),
        ('lgb', LGBMRegressor(random_state=42))
    ]
    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        passthrough=False,
        n_jobs=-1
    )

    # 3. 外层分组交叉验证
    cv = GroupKFold(n_splits=5)
    scores = cross_val_score(
        stack,
        X, y,
        groups=groups,
        cv=cv,
        scoring='r2',
        n_jobs=-1
    )
    print("Stacking R² scores per fold:", scores)
    print(f"Mean R²: {scores.mean():.4f} | Std  R²: {scores.std():.4f}")

    # 4. 在全量数据上训练并保存模型
    stack.fit(X, y)
    os.makedirs("models", exist_ok=True)
    joblib.dump(stack, "models/stacking_model.pkl")
    print("Saved trained Stacking model to models/stacking_model.pkl")

if __name__ == "__main__":
    main()
