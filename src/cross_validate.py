# src/cross_validate.py
print(">>>> cross_validate.py is running <<<<") 
from src.load_data import load_datasets
from src.feature_extraction import build_feature_matrix
from sklearn.model_selection import GroupKFold, cross_val_score
from xgboost import XGBRegressor

def main():
    # 1. 加载并提取特征
    kin, strain = load_datasets()
    X = build_feature_matrix(kin)
    y = strain['90PercStrain']
    groups = kin['subject']  # 按受试者分组

    # 2. 定义模型和交叉验证器
    model = XGBRegressor(random_state=42)
    cv = GroupKFold(n_splits=5)

    # 3. 交叉验证
    scores = cross_val_score(
        model, X, y,
        groups=groups,
        cv=cv,
        scoring='r2',
        n_jobs=-1
    )

    # 4. 输出结果
    print("GroupKFold R² scores for each fold:", scores)
    print(f"Mean R²: {scores.mean():.4f} | Std R²: {scores.std():.4f}")

if __name__ == "__main__":
    main()
