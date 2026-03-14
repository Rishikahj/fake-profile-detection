# ============================================
# Fake Profile Detection - Model Training
# ============================================
# This script trains a Random Forest Classifier
# to detect fake social media profiles and saves
# the trained model and performance report.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ---- Step 1: Load Dataset ----
# Load the profile dataset from CSV file
data = pd.read_csv("dataset/profiles.csv")

# ---- Step 2: Prepare Features and Target ----
# X = input features (all columns except 'fake')
X = data.drop("fake", axis=1)

# y = target label (0 = Real, 1 = Fake)
y = data["fake"]

# ---- Step 3: Split Data ----
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- Step 4: Train the Model ----
# Using Random Forest Classifier for better accuracy
# and ability to handle multiple features
model = RandomForestClassifier()
model.fit(X_train, y_train)

# ---- Step 5: Evaluate the Model ----
# Predict on test data
y_pred = model.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)

# Generate detailed classification report
report = classification_report(y_test, y_pred)

# Get feature importance scores
# (shows which features matter most for detection)
importance = model.feature_importances_

# ---- Step 6: Save Results to File ----
with open("model/model_report.txt", "w") as f:
    f.write("Model Accuracy:\n")
    f.write(str(accuracy))
    f.write("\n\nClassification Report:\n")
    f.write(report)
    f.write("\nFeature Importance:\n")
    for i, col in enumerate(X.columns):
        f.write(f"{col}: {round(importance[i],3)}\n")

# ---- Step 7: Save Trained Model ----
# Save model as .pkl file for use in the web app
joblib.dump(model, "model/model.pkl")

print("Model trained successfully!")
print("Report saved in model/model_report.txt")
