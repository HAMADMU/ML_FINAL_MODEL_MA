import numpy as np
import pandas as pd
from pathlib import Path

INPUT_XLSX = Path("Copy of IEEE68bus_vary_wind_demand_and_control.xlsx")
OUT_ML_200 = Path("IEEE68bus_ML_ready_first200.csv.gz")

def main():
    df = pd.read_excel(INPUT_XLSX, engine="openpyxl", nrows=200)
    print("Loaded (first 200):", df.shape, flush=True)

    df = df.replace(r"^\s*$", np.nan, regex=True)
    df = df.dropna(axis=1, how="all")

    df = df.copy()
    df["unstable"] = (df["RPRMM_0Hz"] >= 0).astype("int8")

    label_like_cols = [c for c in df.columns if c == "RPRMM_0Hz" or str(c).startswith("DRLDM")]
    ml_df = df.drop(columns=label_like_cols, errors="ignore")

    ml_df.to_csv(OUT_ML_200, index=False, compression="gzip")
    print("Saved:", OUT_ML_200.resolve(), flush=True)
    print("Class balance:", ml_df["unstable"].value_counts().to_dict(), flush=True)

if __name__ == "__main__":
    main()