import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 12000
classes = rng.choice(["Normal", "Warning", "Critical"], size=N, p=[0.55, 0.30, 0.15])
rows = []
for status in classes:
    if status == "Normal":
        temperature, vibration, current, rpm = rng.normal(43,4.5), rng.normal(.22,.06), rng.normal(1.15,.18), rng.normal(1480,25)
        operating_hours, machine_age = rng.uniform(100,2500), rng.uniform(.5,5)
    elif status == "Warning":
        temperature, vibration, current, rpm = rng.normal(59,6), rng.normal(.52,.12), rng.normal(1.75,.28), rng.normal(1400,45)
        operating_hours, machine_age = rng.uniform(1200,6000), rng.uniform(2,8)
    else:
        temperature, vibration, current, rpm = rng.normal(74,7), rng.normal(.88,.16), rng.normal(2.55,.42), rng.normal(1280,70)
        operating_hours, machine_age = rng.uniform(3500,10000), rng.uniform(5,12)
    rows.append([temperature,vibration,current,rpm,operating_hours,machine_age,status])
df = pd.DataFrame(rows, columns=["temperature","vibration","current","rpm","operating_hours","machine_age","status"])
df["temperature"] = df["temperature"].clip(25,100).round(2)
df["vibration"] = df["vibration"].clip(.03,1.5).round(3)
df["current"] = df["current"].clip(.3,4.5).round(2)
df["rpm"] = df["rpm"].clip(900,1600).round().astype(int)
df["operating_hours"] = df["operating_hours"].round(1)
df["machine_age"] = df["machine_age"].round(2)
os.makedirs("data", exist_ok=True)
df.to_csv("data/machine_data.csv", index=False)
print(f"Generated {len(df):,} records.")
print(df["status"].value_counts())
print("Saved to data/machine_data.csv")
