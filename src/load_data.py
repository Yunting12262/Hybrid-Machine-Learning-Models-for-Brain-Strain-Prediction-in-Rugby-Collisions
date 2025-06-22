import pandas as pd
import os

def load_datasets():
    # 取脚本所在目录，然后上一级就是项目根，再加 data
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data")

    kin_path = os.path.join(data_dir, "Rugby_data_1701.xlsx")
    str_path = os.path.join(data_dir, "Rugby_simulation_1701.xlsx")

    print("Loading from:", kin_path, str_path)
    kinematics = pd.read_excel(kin_path)
    strain    = pd.read_excel(str_path)
    return kinematics, strain

if __name__ == "__main__":
    X, y = load_datasets()
    print("Loaded shapes:", X.shape, y.shape)
