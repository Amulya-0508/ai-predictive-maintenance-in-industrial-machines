import joblib, pandas as pd
model = joblib.load("models/predictive_maintenance_model.pkl")
sample = pd.DataFrame([{"temperature":68.0,"vibration":0.82,"current":2.35,"rpm":1330,"operating_hours":4200,"machine_age":6.2}])
prediction = model.predict(sample)[0]
print("Machine condition:", prediction)
for cls,p in zip(model.classes_, model.predict_proba(sample)[0]): print(f"{cls}: {p*100:.2f}%")
