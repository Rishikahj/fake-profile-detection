from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__, template_folder="webapp/templates")

# Load trained model
model = joblib.load("model/model.pkl")

# Read accuracy from report file
accuracy = "Not available"

try:
    with open("model/model_report.txt", "r") as f:
        lines = f.readlines()
        accuracy = lines[1].strip()
except:
    pass


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    followers = int(request.form['followers'])
    following = int(request.form['following'])
    posts = int(request.form['posts'])
    bio_length = int(request.form['bio_length'])
    profile_picture = int(request.form['profile_picture'])
    username_length = int(request.form['username_length'])

    features = np.array([[followers, following, posts, bio_length, profile_picture, username_length]])

    prediction = model.predict(features)

    probability = model.predict_proba(features)
    confidence = round(max(probability[0]) * 100, 2)

    if prediction[0] == 1:
        result = "Fake Profile"
    else:
        result = "Real Profile"

    # Risk Level
    if confidence > 80:
        risk = "HIGH"
    elif confidence > 60:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    # Reasons
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

    return render_template(
        "result.html",
        result=result,
        confidence=confidence,
        risk=risk,
        reasons=reasons,
        accuracy=accuracy
    )


if __name__ == "__main__":
    app.run(debug=True)