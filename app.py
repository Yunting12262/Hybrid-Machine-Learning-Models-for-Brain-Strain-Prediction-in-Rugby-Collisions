# app.py

import streamlit as st
import numpy as np
import joblib

# 页面标题
st.set_page_config(page_title="Brain Strain Predictor Demo", layout="wide")
st.title("🏉 Brain Strain Predictor Demo")

# 说明文字
st.markdown("""
这是一个简化版 Demo，只使用以下五个特征：  
- **Peak Linear Acceleration (PLA)**  
- **Peak Rotational Velocity (PRV)**  
- **Peak Rotational Acceleration (PRA)**  
- **FFT Peak Frequency**  
- **FFT Energy**  

输入这些值后，点击 **Predict** 即可获得风险分类和（高风险时）应变回归预测。
""")

# 侧边栏输入
st.sidebar.header("Input Sensor Features")
PLA    = st.sidebar.number_input("Peak Linear Acceleration (g)",     min_value=0.0, value=30.0, step=1.0)
PRV    = st.sidebar.number_input("Peak Rotational Velocity (rad/s)",  min_value=0.0, value=15.0, step=1.0)
PRA    = st.sidebar.number_input("Peak Rotational Acceleration (rad/s²)", min_value=0.0, value=2000.0, step=100.0)
FFT_pf = st.sidebar.number_input("FFT Peak Frequency (Hz)",           min_value=0.0, value=50.0, step=1.0)
FFT_en = st.sidebar.number_input("FFT Energy",                       min_value=0.0, value=1e5, step=1e4)

# 加载 demo 模型
@st.cache_resource
def load_models():
    clf = joblib.load("models/demo_classifier.pkl")
    reg = joblib.load("models/demo_regressor.pkl")
    return clf, reg

clf, reg = load_models()

# 点击预测按钮
if st.sidebar.button("Predict"):
    # 构造输入，必须是二维列表
    X_user = [[PLA, PRV, PRA, FFT_pf, FFT_en]]
    
    # 风险分类
    risk_proba = float(clf.predict_proba(X_user)[0, 1])
    risk_label = "🔴 High Risk" if risk_proba > 0.5 else "🟢 Low Risk"
    
    st.subheader("🚨 Injury Risk Prediction")
    st.write(f"**Risk Level:** {risk_label}")
    st.write(f"**Probability of High Risk:** {risk_proba:.2f}")
    
    # 高风险时的应变回归
    if risk_proba > 0.5:
        strain_pred = float(reg.predict(X_user)[0])
        st.subheader("💡 Estimated Brain Strain (MPS90)")
        st.write(f"**Predicted MPS90:** {strain_pred:.3f}")
    else:
        st.info("Predicted low risk; regression not performed.")

# 底部署名
st.markdown("---")
st.caption("Built by Yunting for MSc Design Engineering Project")
