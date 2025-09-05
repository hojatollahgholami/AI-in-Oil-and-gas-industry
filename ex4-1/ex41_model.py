#!pip install sklearn shap plotly streamlit
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils import class_weight
import shap

file_path = "predictive_maintenance_data.csv"
df = pd.read_csv(file_path)

# Define Features
sensor_features = ["Temperature", "Pressure", "Vibration", "Humidity", "Flow Rate"]
target = "Failure Occurred"

df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Handle missing values (if any)
#df.fillna(df.median(), inplace=True)

# Feature Scaling
scaler = StandardScaler()
df[sensor_features] = scaler.fit_transform(df[sensor_features])

# Visualizing Data Distributions
plt.figure(figsize=(12, 8))
for i, feature in enumerate(sensor_features):
        plt.subplot(3, 2, i + 1)
        sns.histplot(df[sensor_features], bins=30, kde=True)
        plt.title(f"Distribution of {sensor_features}")
plt.tight_layout()
plt.show()

    
# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df[sensor_features].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

df.to_csv("cleaned_sensor_data.csv", index=False)
    
    
# Train Machine Learning Models
X = df[sensor_features]
y = df[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression Model
log_reg = LogisticRegression(max_iter=500)
#weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
#log_reg = LogisticRegression(max_iter=500, class_weight={0: weights[0], 1: weights[1]})
log_reg.fit(X_train, y_train)
log_reg_preds = log_reg.predict(X_test)
log_reg_acc = accuracy_score(y_test, log_reg_preds)

# Train Random Forest Model use hyperparameters 
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)

#param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
#grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
#grid_search.fit(X_train, y_train)
#best_rf = grid_search.best_estimator_

# Train XGBoost Model
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_preds)

# Evaluate Models
preds = log_reg.predict(X_test)
acc = accuracy_score(y_test, preds)
cm = confusion_matrix(y_test, preds)
    
print(f"=== {'Logistic Regression'} Evaluation ===")
print(f"Accuracy: {acc:.4f}")
print("Classification Report:", classification_report(y_test, preds))
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Failure", "Failure"], yticklabels=["No Failure", "Failure"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"{'Logistic Regression'} Confusion Matrix")
plt.show()

#######
preds = rf_model.predict(X_test)
acc = accuracy_score(y_test, preds)
cm = confusion_matrix(y_test, preds)
    
print(f"=== {'Random Forest'} Evaluation ===")
print(f"Accuracy: {acc:.4f}")
print("Classification Report:", classification_report(y_test, preds))
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Failure", "Failure"], yticklabels=["No Failure", "Failure"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"{'Random Forest'} Confusion Matrix")
plt.show()

#######
preds = xgb_model.predict(X_test)
acc = accuracy_score(y_test, preds)
cm = confusion_matrix(y_test, preds)
print(f"=== {'XGBoost'} Evaluation ===")
print(f"Accuracy: {acc:.4f}")
print("Classification Report:", classification_report(y_test, preds))
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Failure", "Failure"], yticklabels=["No Failure", "Failure"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"{'XGBoost'} Confusion Matrix")
plt.show()
# Feature Importance using SHAP for "Random Forest"
explainer = shap.Explainer(rf_model, X_test)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar", show=True)

# Failure Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="Failure Occurred", data=df, palette="coolwarm")
plt.title("Failure Occurrence Distribution")
plt.xlabel("Failure Occurred (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.xticks(ticks=[0, 1], labels=["No Failure", "Failure"])
plt.show()

# Sensor Readings vs. Failures
plt.figure(figsize=(12, 6))
for i, feature in enumerate(sensor_features):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(x="Failure Occurred", y=feature, data=df, palette="coolwarm")
    plt.title(f"{feature} vs. Failure")
plt.tight_layout()
plt.show()    

