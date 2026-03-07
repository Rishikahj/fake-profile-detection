# Fake Profile Detection System – ML-Based Social Media Analysis

Live Demo: https://fake-profile-detection-huv4.onrender.com

## Project Overview

This project detects potentially fake social media profiles using machine learning and behavioral analysis. The system analyzes profile characteristics such as followers, following count, posting activity, profile completeness, and bio information to classify accounts as **Real** or **Fake**.

The application also provides **confidence score, risk level, and behavioral explanations** for the prediction.

---

## Features

• Machine Learning based fake profile detection
• Behavioral analysis of profile characteristics
• Confidence score for prediction
• Risk level classification (Low / Medium / High)
• Explanation of suspicious indicators
• Web interface for analyzing profiles

---

## Technologies Used

* Python
* Scikit-learn
* Flask
* Pandas
* NumPy
* HTML / CSS

---

## Machine Learning Model

The system uses a **Random Forest Classifier** to detect fake profiles based on behavioral features.

### Features Used

* Followers count
* Following count
* Number of posts
* Bio length
* Profile picture presence
* Username length

---

## Project Structure

```
fake-profile-detection
│
├── dataset
│   └── profiles.csv
│
├── model
│   ├── model.pkl
│   └── model_report.txt
│
├── webapp
│   └── templates
│       ├── index.html
│       └── result.html
│
├── screenshots
│   ├── home_page.png
│   └── result_page.png
│
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

---

## Screenshots

### Home Page

![Home Page](screenshots/home_page.png)

### Prediction Result

![Result Page](screenshots/result_page.png)

---

## How to Run the Project

1. Install dependencies

```
pip install -r requirements.txt
```

2. Train the model

```
python train_model.py
```

3. Run the web application

```
python app.py
```

4. Open in browser

```
http://127.0.0.1:5000
```

---

## Limitations

• The system currently uses simulated profile data rather than real-time social media APIs.
• Predictions indicate suspicious behavior patterns but do not guarantee that an account is fake.

---
Live Demo: https://fake-profile-detection-huv4.onrender.com

