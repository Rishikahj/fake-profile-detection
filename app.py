# ============================================
# Fake Profile Detection - Flask Web App
# ============================================
# This is the main web application file.
# It loads the trained ML model and serves
# predictions through a web interface.

from flask import Flask, render_template, request
import joblib
import numpy as np

# ---- Initialize Flask App ----
# Tell Flask where to find the HTML templates
app = Flask(__name__, template_folder="webapp/templates")

# ---- Load Trained Model ----
# Load the pre-trained Random Forest model from disk
model = joblib.load("model/model.pkl")

# ---- Read Model Accuracy from Report ----
# Try to read accuracy from the saved report file
accuracy = "Not available"
try:
    with open("model/model_report.txt", "r") as f:
        lines = f.readlines()
        accuracy = lines[1].strip()
except:
    pass

# ---- Route 1: Home Page ----
# Displays the input form to the user
@app.route('/')
def home():
    return render_template("index.html")

# ---- Route 2: Prediction ----
# Takes user input, runs ML model, returns result
@app.route('/predict', methods=['POST'])
def predict():

    # ---- Step 1: Get User Input from Form ----
    followers = int(request.form['followers'])
    following = int(request.form['following'])
    posts = int(request.form['posts'])
    bio_length = int(request.form['bio_length'])
    profile_picture = int(request.form['profile_picture'])
    username_length = int(request.form['username_length'])

    # ---- Step 2: Prepare Input for Model ----
    # Convert input values into a numpy array for prediction
    features = np.array([[followers, following, posts, bio_length, profile_picture, username_length]])

    # ---- Step 3: Make Prediction ----
    prediction = model.predict(features)

    # Get probability scores for confidence calculation
    probability = model.predict_proba(features)

    # Calculate confidence as percentage
    confidence = round(max(probability[0]) * 100, 2)

    # ---- Step 4: Determine Result ----
    # 1 = Fake Profile, 0 = Real Profile
    if prediction[0] == 1:
        result = "Fake Profile"
    else:
        result = "Real Profile"

    # ---- Step 5: Calculate Risk Level ----
    # Based on confidence score
    if confidence > 80:
        risk = "HIGH"
    elif confidence > 60:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    # ---- Step 6: Generate Behavioral Reasons ----
    # Explain why the profile is flagged as suspicious
    reasons = []
    if followers < 50:
        reasons.append("Very low followers")
    if following > 800:
        reasons.append("Following too many accounts")
    if posts < 5:
        reasons.append("Very few posts")
    if profile_picture == 0:
        reasons.append("No profile picture")
    if bio_length < 5:
        reasons.append("Very short bio")

    # ---- Step 7: Send Results to Result Page ----
    return render_template(
        "result.html",
        result=result,
        confidence=confidence,
        risk=risk,
        reasons=reasons,
        accuracy=accuracy
    )

# ---- Run the App ----
if __name__ == "__main__":
    app.run(debug=True)
