# Diabetes Prediction Model (One Page)
# Author: Ankush Singh (Educational Project)
# Dataset: Pima Indians Diabetes (Kaggle)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib

# 1. Load data
df = pd.read_csv("diabetes.csv")

# 2. Replace invalid zeros with NaN for specific columns
cols = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
for c in cols:
    df[c] = df[c].replace(0, np.nan)

# 3. Fill missing values using median
imp = SimpleImputer(strategy="median")
df[cols] = imp.fit_transform(df[cols])

# 4. Split features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 5. Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6. Scale numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 8. Evaluate model
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]
print("✅ Accuracy:", round(accuracy_score(y_test, y_pred),3))
print("✅ ROC-AUC:", round(roc_auc_score(y_test, y_prob),3))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 9. Save model + scaler
joblib.dump(model, "diabetes_model.joblib")
joblib.dump(scaler, "diabetes_scaler.joblib")

# 10. Predict for new user
new_data = pd.DataFrame([{
    "Pregnancies": 2, "Glucose": 120, "BloodPressure": 70,
    "SkinThickness": 20, "Insulin": 79, "BMI": 25.0,
    "DiabetesPedigreeFunction": 0.5, "Age": 28
}])
new_scaled = scaler.transform(new_data)
pred = model.predict(new_scaled)[0]
prob = model.predict_proba(new_scaled)[0,1]
print(f"\nPrediction: {'Diabetic' if pred==1 else 'Not Diabetic'} (Prob: {prob:.2f})")
