import numpy as np          #Library module for numbers
import pandas as pd         #Library module for reading data
from pathlib import Path    #library class for integrating dataset

INPUT_XLSX = Path("Copy of IEEE68bus_vary_wind_demand_and_control.xlsx")            #declare input file name
OUT_ML = Path("IEEE68bus_ML_ready_risky.csv.gz")                                    #declare output file name 
RISK_THRESHOLD = -0.01  # risky if RPRMM_0Hz >= -0.01

def main():
    if not INPUT_XLSX.exists():
        raise FileNotFoundError(f"Excel not found: {INPUT_XLSX.resolve()}")             #raise: stop the program and show/print error
                                                                                        #resolve in this situation returns the full path for the spotted error  
    print("Reading Excel (may take time)...", flush=True)
    df = pd.read_excel(INPUT_XLSX, engine="openpyxl") # variable named df.. 
    print("Loaded:", df.shape, flush=True)    # prints shape of array

    # Convert blank/whitespace cells into NaN
    df = df.replace(r"^\s*$", np.nan, regex=True)                #replace string, regex matches all empty spaces

    # ----------------------------
    # 1) Drop fully-null columns
    # ----------------------------
    all_null_cols = df.columns[df.isna().all()].tolist()                  #df.isna(): true/false values, df.isna().all():are all values in the column true?, tolist(): converts to normal python list 
    print("Fully-null columns removed:", len(all_null_cols), flush=True)  #len(): number of items in container
    if all_null_cols:
        df = df.drop(columns=all_null_cols)                               #df.drop: removes specified columns

    # -----------------------------------
    # 2) Drop columns that are all zeros
    # -----------------------------------
    numeric_df = df.select_dtypes(include=[np.number])                      #df.select_dtypes: filter based on data types, include=[np.number]: include only columns with numeric data
    all_zero_cols = numeric_df.columns[(numeric_df == 0).all()].tolist()    #compare all numeric values in each columns with zero, 
    print("Fully-zero columns removed:", len(all_zero_cols), flush=True)    
    if all_zero_cols:
        df = df.drop(columns=all_zero_cols)                                 #if all values in the column = 0, remove the column

    # ----------------------------
    # Create 'risky' label
    # ----------------------------
    if "RPRMM_0Hz" not in df.columns:
        raise KeyError("Column 'RPRMM_0Hz' not found in Excel.")

    df = df.copy()                                                      #duplicate so original is not affected
    df["risky"] = (df["RPRMM_0Hz"] >= RISK_THRESHOLD).astype("int8")    #astype(): converts true/false to 1/0, df[risky]: assigns result to a new column

    r = df["RPRMM_0Hz"]
    print(f"RPRMM range: min={r.min():.6f}, max={r.max():.6f}, mean={r.mean():.6f}", flush=True)    #f: allows to insert variables into the string, :.6f: 6 decimal places
    print("Risk threshold:", RISK_THRESHOLD, flush=True)
    print("Risky label counts:",
          df["risky"].value_counts().rename({0:"not_risky", 1:"risky"}).to_dict(),          #value_counts(): counts how many times each value appears, rename(): replaces numeric labels with readable names
          flush=True)                                                                       #to_dict(): converts results into python dictionary

    # Prevent leakage
    leak_cols = [c for c in df.columns
                 if c == "RPRMM_0Hz" or str(c).startswith("DRLDM")]        #loops through every column name in the dataset, startswitch(x): does column begin with x
    ml_df = df.drop(columns=leak_cols, errors="ignore")                     #remove the specified column

    # Save compressed ML-ready dataset
    ml_df.to_csv(OUT_ML, index=False, compression="gzip")           #to.csv(): export this dataframe as a csv file, compress using GZIP
    print("Saved:", OUT_ML.resolve(), flush=True)
    print("Final ML-ready shape:", ml_df.shape, flush=True)
 

if __name__ == "__main__":
    main()                                                          #only run main if file is executed directly