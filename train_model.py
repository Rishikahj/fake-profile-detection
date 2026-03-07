import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
data = pd.read_csv("dataset/profiles.csv")

# Features
X = data.drop("fake", axis=1)

# Target
y = data["fake"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Classification report
report = classification_report(y_test, y_pred)

# Feature importance
importance = model.feature_importances_

# Save results to file
with open("model/model_report.txt", "w") as f:
    f.write("Model Accuracy:\n")
    f.write(str(accuracy))
    f.write("\n\nClassification Report:\n")
    f.write(report)
    f.write("\nFeature Importance:\n")

    for i, col in enumerate(X.columns):
        f.write(f"{col}: {round(importance[i],3)}\n")

# Save model
joblib.dump(model, "model/model.pkl")

print("Model trained successfully!")
print("Report saved in model/model_report.txt")