import argparse
import pandas as pd
import numpy as np
import miceforest as mice

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="HW1_CleanedDataset.csv")
    parser.add_argument("--bmi_col", default="bmi")  # case-insensitive
    parser.add_argument("--mice_iters", type=int, default=10)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    # ---- 1) BMI outliers -> set to NaN (IQR) ----
    bmi_col = next((c for c in df.columns if str(c).lower()==str(args.bmi_col).lower()), None)
    if bmi_col is None:
        raise ValueError("BMI column not found. Use --bmi_col to specify.")
    bmi_series = pd.to_numeric(df[bmi_col], errors="coerce")
    q1, q3 = bmi_series.quantile(0.25), bmi_series.quantile(0.75)
    iqr = q3 - q1
    lb, ub = q1 - 1.5*iqr, q3 + 1.5*iqr
    df.loc[(bmi_series < lb) | (bmi_series > ub), bmi_col] = np.nan

    # ---- 2) One-hot encode categoricals ----
    df_enc = pd.get_dummies(df, drop_first=False, dtype="int")

    # ---- 3) MICE imputation (compatible with miceforest 6.x) ----
    kernel = mice.ImputationKernel(df_enc)

    kernel.mice(args.mice_iters)

    try:
        df_imp = kernel.complete_data(0)   
    except TypeError:
        df_imp = kernel.complete_data()    


    # ---- 4) Min–Max normalization (all numeric) ----
    df_norm = (df_imp - df_imp.min()) / (df_imp.max() - df_imp.min())

    # ---- 5) Save ----
    df_norm.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
