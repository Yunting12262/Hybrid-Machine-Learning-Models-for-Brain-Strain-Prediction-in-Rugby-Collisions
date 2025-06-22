import pandas as pd
from src.load_data import load_datasets
from src.feature_extraction import build_feature_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


# 加载
kin, strain = load_datasets()
X = build_feature_matrix(kin)
y = strain['90PercStrain']

# 划分
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练
model = XGBRegressor(random_state=42)
model.fit(X_tr, y_tr)


# 评估
print("Test R2:", model.score(X_te, y_te))