# src/analysis_shap.py

import os
import shap
import matplotlib.pyplot as plt
from src.load_data import load_datasets
from src.feature_extraction import build_feature_matrix
from xgboost import XGBRegressor

def main():
    kin, strain = load_datasets()
    X = build_feature_matrix(kin)
    y = strain['90PercStrain']

    model = XGBRegressor(random_state=42)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    # 确保 picture/ 目录存在
    os.makedirs("picture", exist_ok=True)

    # 1) 平均绝对 SHAP 值条形图
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X,
        plot_type="bar",
        max_display=15,
        show=False
    )
    plt.title("SHAP Feature Importance (Mean |SHAP|)")
    plt.tight_layout()
    plt.savefig("picture/shap_bar.png", dpi=200)
    plt.close()

    # 2) 蜂群图（SHAP beeswarm）
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X,
        plot_type="dot",
        max_display=15,
        show=False
    )
    plt.title("SHAP Beeswarm Plot")
    plt.tight_layout()
    plt.savefig("picture/shap_beeswarm.png", dpi=200)
    plt.close()

    print("Saved shap_bar.png and shap_beeswarm.png to picture/")

if __name__ == "__main__":
    main()
