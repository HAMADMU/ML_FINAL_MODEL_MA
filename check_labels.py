import pandas as pd

df = pd.read_excel("Copy of IEEE68bus_vary_wind_demand_and_control.xlsx", engine="openpyxl")

r = df["RPRMM_0Hz"]
print("RPRMM min:", r.min())
print("RPRMM max:", r.max())
print("RPRMM mean:", r.mean())
print("count >= 0:", (r >= 0).sum())

d = df["DRLDM"]
print("\nDRLDM min:", d.min())
print("DRLDM max:", d.max())
print("count < 0.05:", (d < 0.05).sum())
print("count < 0.03:", (d < 0.03).sum())