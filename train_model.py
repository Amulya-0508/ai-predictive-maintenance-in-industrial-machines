import os, json, joblib, pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

os.makedirs("models", exist_ok=True)
df = pd.read_csv("data/machine_data.csv")
features = ["temperature","vibration","current","rpm","operating_hours","machine_age"]
X, y = df[features], df["status"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,stratify=y)
model = RandomForestClassifier(n_estimators=250,random_state=42,class_weight="balanced",n_jobs=-1)
model.fit(X_train,y_train)
pred = model.predict(X_test)
accuracy = accuracy_score(y_test,pred)
print(f"Accuracy: {accuracy:.4f}\n")
print(classification_report(y_test,pred))
joblib.dump(model,"models/predictive_maintenance_model.pkl")
with open("models/model_metadata.json","w") as f: json.dump({"features":features,"classes":list(model.classes_),"accuracy":float(accuracy)},f,indent=2)
cm = confusion_matrix(y_test,pred,labels=model.classes_)
ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_).plot()
plt.title("Predictive Maintenance - Confusion Matrix")
plt.tight_layout(); plt.savefig("models/confusion_matrix.png",dpi=160); plt.close()
print("Model saved to models/predictive_maintenance_model.pkl")
